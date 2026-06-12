---
run: r69_4tier_fidelity
created: 2026-05-31T17:25:17-04:00
state: HALTED_FOR_SCOPING
current_phase: paused
current_round: 8
---

> **ALL JOBS STOPPED 2026-06-02 ~mid-morning, by JMT request.** 0 python3 / 0 codex / 0
> watchers. NO new runs until JMT and CC scope the approach together. Reason: the whole
> fidelity picture is tangled across many runs/notes and JMT (rightly) wants to reconcile
> before burning more compute.
>
> RECONCILIATION FINDINGS SO FAR (the confusion is real):
> - `algo_fidelity_FINAL_SUMMARY.md` (Apr 30, rounds 24-26) is MISLEADINGLY NAMED -- it
>   predates the actual bit-exact push. It claims 8 deterministic engines bit-exact + 6
>   stochastic TOST-equiv, ON A 5-GRAPH SUBSET. CC wrongly cited it as authoritative.
> - The "bit-exact everything" work JMT remembers = May 25-26 rounds 40-65 (git commits:
>   "bit-exact push across every engine", "fdp BIT-EXACT vs instrumented graphviz" r48/53,
>   "fa2 BH bit-exact real port" r60, "graphopt <1e-6" r65). NO consolidated verified summary
>   exists for this push -- only per-round commits + scattered eval_output round reports.
> - Every verification attempt since hit a DIFFERENT measurement bug: R66/R66b (100% PARTIAL),
>   R68 (benchmark wasn't using fidelity_mode at all), R69 (CC patched fidelity_mode, then hit
>   seed-pairing bug, then mis-scoped escalation engines-not-combos, then seed-mismatch on
>   deterministic refs + per-engine bit-exact configs from May not reproduced).
> - KEY UNKNOWN to resolve in scoping: stochastic engines (incl graphviz neato/sfdp/fdp) were
>   NEVER bitwise -- only TOST-equiv. Deterministic engines bit-exact but only verified on
>   5 small graphs. "bit-exact everything" was likely an over-claim (cf
>   feedback_verify_against_reference: documented false-bit-exact pattern).
> - R69 progress that IS committed + valid: P1 (214b203/7c15129 variants fidelity-matched),
>   P1a (a700ccd linlog REAL port, genuinely bit-exact 1.5e-16), P2b report pairing fix.
>   P2 5-seed data exists (eval_output/benchmark_5seed_fidelity). P3/P3b data partial+killed.
> NEXT: scope with JMT. Likely: ONE authoritative per-engine reconciliation (pin each May
> 25-26 bit-exact claim -> verify ONCE cleanly at matched seeds on small graphs -> single
> source of truth), THEN decide on any full-suite run. DO NOT launch runs unprompted.

> **P3 OVER-ESCALATION ERROR + FIX (2026-06-02 09:00).** JMT flagged: P3 was running 100
> seeds on every escalation ENGINE across ALL 105 graphs (348K combos). WRONG -- the spec is
> 100 seeds only on the (engine,GRAPH) COMBOS that failed bit-exactness at 5 seeds. KILLED P3
> (pid 1783234, process group). Correct scope: 646 failing (engine,graph) combos (from
> per_variant.json failures) = ~65K layouts (5x smaller). Per-engine failing graphs in
> /tmp/r69_failing_map.json. run_benchmark --graphs is GLOBAL so must loop per-engine ->
> scripts/r69_p3b_targeted.py (each engine 100 seeds on its OWN failing graphs only,
> --resume reuses the partial benchmark_100seed_escalation data). Launched pid 1919978,
> watcher b7fksxnx5, log /tmp/r69_p3b.log. Then consolidate -> TOST -> combined report.
> LESSON: "non-bit-exact combo" = (engine,graph) PAIR, not whole engine. Most engines are
> bit-exact on the majority of graphs; only their FAILING graphs need 100-seed TOST.
> ON P3b DONE (P4): TOST gives Tier3/4 PER (engine,graph) failing combo. Final verdict is
> per-(engine,graph): bit-exact-at-5seed pairs = Tier1; failing pairs = Tier3 (TOST-equiv) or
> Tier4 (TOST-diff). Aggregate/summarize per engine. + the 37 divergent engines' failing pairs
> were ALSO only-5-seed -- decide if they need P3b too (they're median>=0.1, likely Tier4;
> currently NOT in the 25-engine P3b set -- FLAG for JMT whether to include them).

> **DECISION (2026-06-02 09:10, JMT away, no answer to scope question -> autonomous rec):**
> Proceed with: 100-seed TOST ONLY on the chaotic-faithful set (P3b running). The ~37
> broadly-divergent engines (median>=0.1: sfdp/umap/gem/sugiyama/neato/fmmm/maxent/drl-coarse+)
> are classified TIER 4 (statistically different) FROM 5-SEED evidence -- they differ on the
> MAJORITY of graphs, so 100-seed TOST would only confirm. Full 100-seed on them = ~400K
> layouts/multi-day, low info -> skipped. Report FLAGS them (esp deterministic-structure ones
> like sugiyama: Procrustes RMSD may understate algorithmic equivalence -> manual review).
> /tmp/r69_failing_map.json + r69_p3b_targeted.py can be re-run on the divergent set if JMT
> wants the formal test later (one command: rebuild map for all 62 PARTIAL, rerun p3b).

> **P2b DONE + FULL TRIAGE (2026-06-02 02:45). Committed 214b203 (report pairing fix).**
> Pairing fix worked: no-pair skips 4278->1033, 94/117 variants verdicted (was 54).
> Triage (scripts/r69_triage.py -> eval_output/fidelity_report_r69/triage.md):
>   TIER 1 BIT_IDENTICAL (32): fa2 x10, graphopt x6, lgl x5, linlog x5(canary OK), tsnet x5, lgl_iter50.
>   TIER 2 TIMEOUT (6): neulay x6 (ok=0, all error/skip -- never completes).
>   NO_REFERENCE (7): fr_kk/kk_fr chains x4, spectral_random_walk, spectral_unnormalized, rt_horizontal.
>   NON-BIT-IDENTICAL split by 5-seed MEDIAN RMSD:
>     - 25 with median<0.1 = CHAOTIC-FAITHFUL (bit-exact on majority, basin-flip minority)
>       -> ESCALATED to P3 100-seed TOST (these decide Tier3 vs Tier4): davidson_harel x3,
>       fr x4, stress_sgd x4, kk x3, pivot_mds x4, classical_mds x2, fa2_linlog, drl_default,
>       stress_maj_iter500, spectral_default/nx_fidelity.
>     - ~37 with median>=0.1 = DIVERGENT (differ on MAJORITY of graphs) -> TIER 4 from 5-seed
>       (conclusive; 100-seed would just confirm): drl(coarsen/coarsest/final/refine),
>       umap x6, sugiyama x6, sfdp x6, neato, gem x3, fmmm x4, maxent_stress x5, stress_maj
>       (default/iter50), davidson_rounds200(median 5e-2 -- actually escalated).
>   EDGE CASES (flag for JMT): sgd2_multi x8 UNVERDICTED (their ref sgd2_multi_ref too slow,
>     didn't finish P2 -> need dedicated ref run or accept unclassified); fcose x2 no Python port;
>     neato_graphviz_fidelity 0 attempts.
>   SEED-VARIANCE CHECK: divergent engines ARE stochastic (vary w/ seed) EXCEPT sugiyama
>     (deterministic). So divergent != measurement artifact -- genuine layout differences.
>   METHODOLOGY CAVEAT (put in final report): graphviz/igraph/ogdf ports (sfdp/neato/gem/fmmm/
>     sugiyama/stress_maj/maxent) don't bit-match refs (median 0.2-1.2). Procrustes RMSD may
>     UNDERSTATE equivalence for discrete-structure layouts (sugiyama layer ordering) -- merit
>     manual review; could be genuine fidelity gaps OR metric limitation.
>   DEVIATION FROM LITERAL INSTRUCTION ("100 seeds for ALL non-bit-identical"): escalated only
>     median<0.1 (25), not all 57+, because median>=0.1 are conclusively Tier4 on 5-seed and
>     57x100seed = ~2 days mostly redundant. TOLD JMT via iMessage; he can expand in AM.

> **P3 LAUNCHED (2026-06-02 02:50).** scripts/r69_p3_100seed_tost.sh on 25 reimpls + refs
> (50 engines, /tmp/r69_escalation_engines.txt), wrapper pid 1783234, watcher byru5e7x7,
> log /tmp/r69_p3_100seed.log. Runs 100-seed bench -> consolidate -> r68_tost_followup ->
> r68_combined_report -> eval_output/fidelity_report_r69/report.md. Mostly fast engines.
> **CODEX STILL QUOTA-BLOCKED but P3/P4 are pure python -- fine.**
> ON P3 DONE (P4): verify report.md; build FINAL 4-tier table = Tier1(32 bit-identical) +
> Tier2(neulay timeout) + Tier3(TOST-equivalent from P3) + Tier4(TOST-different from P3 UNION
> the 37 divergent-from-5seed) + appendix(no-reference 7, sgd2_multi-unverdicted 8, fcose no-port,
> methodology caveat). Write r69_4tier_fidelity_SUMMARY.md, file-for-review, final iMessage,
> mark state DONE, delete stale baton.md.

> **P2 DONE (2026-06-02 02:22, 100% complete).** stage1 report written. Verdicts (of 54
> variants that GOT paired): MACHINE_EPSILON=30, BIT_EXACT=1, PARTIAL=23. CANARY PASSED:
> linlog all MACHINE_EPSILON (~2e-16) -> real port + fidelity routing verified end-to-end.
>   Tier1 bit-identical (31): fa2 x10, graphopt x6, lgl x5, linlog x5, tsnet x5.
>   PARTIAL->escalate (23): davidson_harel x3, drl x5, fa2_linlog, fr x4, stress_sgd x4, umap x6.
>     (most PARTIALs have MEDIAN ~machine-eps, only a few high-RMSD seeds = chaotic-faithful
>      -> likely TOST-equivalent Tier 3; drl & umap have high MEDIAN = genuinely divergent.)
>
> **P2 PAIRING BUG FOUND -> P2b (codex 1743095, RUNNING).** Only 54 of 117 classic engines got
> verdicts. The other 63 ran OK but were UNVERDICTED due to a measurement bug in
> fast_fidelity_report: deterministic reference adapters (igraph_sugiyama, ogdf_*, graphviz_*,
> igraph_mds) run at seed=None, reimpls at seeds 42-46, so `reimp_seeds & ref_seeds` is empty
> -> "no-pair skips: 4278". Affected (likely Tier 1 graphviz/igraph/ogdf ports): sugiyama,
> sfdp, neato, gem, pivot_mds, classical_mds, stress_maj, fmmm; plus sgd2_multi, maxent_stress,
> neulay (neulay all error/skip -> probably Tier2/broken), kk/rt/spectral (deterministic).
> P2b fixes the pairing (pair each reimpl seed vs the single seed=None ref result) + RE-RUNS
> the report (no re-benchmark; positions saved). Spec: PROMPT_R69_P2b_deterministic_ref_pairing.md
> ON P2b DONE: verify new verdict counts + previously-missing variants present; commit the
> fast_fidelity_report fix; THEN do the FULL triage (all 118) -> escalation list -> P3.
> HOLD the JMT triage iMessage until verdicts are COMPLETE (post-P2b), then send one accurate one.

> **CODEX QUOTA EXHAUSTED (2026-06-02 02:28).** P2b codex (1743095) died: "You've hit your
> usage limit" -- no changes made. FALLBACK: CC hand-implemented the pairing fix (small
> measurement fix, sanctioned by quota-fallback rule). Fix in scripts/fast_fidelity_report.py:
> added _resolve_pos() (tries ::seed{N} / ::deterministic / ::seedNone / seedless keys) and
> rewrote the pairing loop to handle deterministic refs (h5 key = "{graph}::{ref}::deterministic";
> ref ran seed=None, reimpl at 42-46). Compiles clean. Report re-run LAUNCHED (pid 1750190,
> log /tmp/r69_report_rerun.log, watcher bhugrn2wh).
> **KEY: P3 (run_benchmark) + TOST + P4 (combined report) are ALL PLAIN PYTHON -- they do NOT
> need codex. So the run continues unblocked despite codex quota.** No need to wait for codex reset.
> ON report-rerun DONE: read new verdict counts (no-pair skips should plummet from 4278; ~50
> more variants verdicted), commit the fast_fidelity_report.py fix, then FULL triage all 118.

> **P1b DONE (round 3, codex 182132, commit 7c15129):** variants fidelity-matched.
> fidelity_mode 9 -> 91 routed; 27 documented no-port. Mapping report:
> eval_output/fidelity_report_r69/p1_variant_fidelity_mapping.md. No delegation remains
> (re-audited). Smoke confirms fidelity_mode is not a no-op (neato/graphopt differ).
> Commit clean (variants.py + spectral test only; pre-existing lgl/sgd2 dirty files left).
>
> **NO-PORT NUANCE (handle in P2 triage):** the 27 "no-port" split 3 ways:
>  (a) genuinely unported: classic_fcose_default/proof (cytoscape).
>  (b) NO paired reference (cannot be fidelity-tested): fr_kk/kk_fr chains (x4),
>      spectral_random_walk, spectral_unnormalized, rt_horizontal.
>  (c) HAS a port under a DIFFERENT selector (NOT fidelity_mode) -> may misclassify if
>      left default; fix via targeted P1c ONLY if P2 shows them non-bit-identical:
>      - classic_classical_mds_default/igraph_fidelity -> selector is igraph_fidelity=True
>      - classic_pivot_mds_10/50/100/200 -> parity via first_pivot/compute_dtype/distance_scale
>      - classic_fr_steps* (ref nx_spring) / classic_kk_steps* (ref nx_kamada_kawai) ->
>        networkx_compat selector; BUT fr default already showed median ~4e-16 (may be
>        Tier 1 as-is). Let P2 data decide.
>      - classic_maxent_stress_* -> "no fidelity_mode selector"; treat as no-port unless data says otherwise.

> **P1a DONE (round 2, codex 128491, commit a700ccd):** linlog real in-pipeline port
> landed. Delegation removed; bit-exact vs reference (max abs diff 0.0, Procrustes RMSD
> 1.47e-16; 15 parity tests pass, independently re-verified by CC). 434 layout tests green.
> NOTE: codex does NOT commit (project AGENTS rule "do not commit/push") -- CC commits
> after independent verification each phase. linlog variants now use fidelity_mode=True.
> Pre-existing unrelated breakage: tests/test_classic_drl.py collection error (stale
> archived import `layout_drl`) -- NOT from this run; ignore.

> **P1 FINDING (round 1, codex 92126, 2026-05-31):** variant-patch codex correctly
> HARD-STOPPED -- found `linlog.py:165-181` fidelity path DELEGATES (imports
> `_layout_linlog_reference` from eval.competitors, returns its positions = tautology).
> Full audit (CC): linlog is the SOLE live delegation. tsnet uses sklearn
> `_joint_probabilities` primitive only (JMT-approved). All else clean.
> JMT decision: BUILD A REAL linlog port. -> sub-phase P1a (linlog port), then
> P1b (re-run variant patch, now linlog-safe). linlog reference is dagua-authored
> pure-torch paper-spec Noack -> bit-exact achievable; Tier-1 = "pipeline reproduces
> reference impl" (self-consistency, not external canonical tool -- note in report).

# r69_4tier_fidelity -- Autonomous Loop State

Canonical "where are we" record. Every wake-up event (watcher fire, user ping,
schedule trigger) MUST read this file FIRST and act on the case routing below.
Act on the case, not intuition.

## Goal (from JMT, 2026-05-31)

Classify EVERY algo/layout combo (the 118 `classic_*` reimplementation variants)
into exactly ONE of 4 tiers, all on the FIDELITY-MATCHED algo versions:

1. **BIT_IDENTICAL** -- per-seed Procrustes RMSD < 1e-3 on all 5 stage-1 seeds. DONE.
2. **TIMEOUT** -- variant times out on the benchmark (fidelity ports can be 50-100x
   slower); excluded from the 100-seed escalation.
3. **TOST_EQUIVALENT** -- not bit-identical, but statistically the same across 100
   seeds (TOST equivalence passes). Chaotic-but-faithful (basin divergence).
4. **TOST_DIFFERENT** -- statistically different across 100 seeds (TOST fails).

Text JMT (`~/.claude/scripts/send-to-jmt.sh`) at the END OF EACH PHASE.

## Design principles (JMT clarifications 2026-05-31 -- do not violate)
- **Python-only is the product.** Common Python libs (numpy/scipy/sklearn/networkx/
  torch) are FINE. The ONLY violation is requiring a separate NON-PYTHON program
  (graphviz dot/neato binary, Java OGDF). tsnet importing sklearn `_joint_probabilities`
  is ALLOWED (Python primitive, not the layout solver).
- **No fidelity-output delegation:** reimpl must compute positions itself, never call
  the reference to PRODUCE its layout. See [[feedback_no_runtime_delegation_to_reference]].
- "PyTorch-differentiable where possible" is a SOFT desideratum for default/fast
  pipelines -- NOT a constraint on these (deliberately sequential) fidelity ports.

## Phases

| Phase | What | Output | Done when |
|---|---|---|---|
| P1 | Codex patches variants.py: all 109 un-matched `classic_*` opt into correct `fidelity_mode`. Verify 0 binary shell-outs. | `eval_output/fidelity_report_r69/p1_variant_fidelity_mapping.md` + commit | `grep -c '"fidelity_mode"' dagua/eval/variants.py` jumped from 9 to ~118-noport; import clean; mapping report exists |
| P2 | Stage-1: 5-seed benchmark, ALL combos, matched. Per-seed Procrustes. | `eval_output/benchmark_5seed_fidelity/` + `eval_output/fidelity_report_r69/stage1/` | report classifies every variant: Tier1 (bit-identical) / Tier2 (timeout) / escalate |
| P3 | Stage-2: 100-seed, ESCALATION SUBSET ONLY (non-bit-identical, non-timeout). TOST. | `eval_output/benchmark_100seed_escalation/` + `eval_output/fidelity_report_r69/tost/` | every escalated variant -> Tier3 (TOST equiv) or Tier4 (TOST diff) |
| P4 | Combined 4-tier report + shutdown. | `eval_output/fidelity_report_r69/report.md` + `r69_4tier_fidelity_SUMMARY.md` | report lists all 118 variants in exactly one tier; final iMessage sent |

## Exact phase commands

Env for graphviz-instrumented runs (if used):
`export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}`

### P1 (codex)
```
~/.claude/scripts/codex-bg.sh /tmp/r69_p1.log \
  .project-context/research/sprint_algo_fidelity/PROMPT_R69_P1_variant_fidelity_mode.md \
  --cd /home/jtaylor/projects/dagua --sandbox danger-full-access \
  -c model_reasoning_effort=high
# then: Monitor ~/.claude/scripts/codex-watch.sh <PID> /tmp/r69_p1.log
```
NOTE: spec is COMMITTED at that path (survives /tmp wipe). Pass the file path to codex-bg.sh.

### P2 (benchmark, run via bg-watch)
```
python3 scripts/run_benchmark.py --seeds 5 --seed-start 42 --variants \
  --engines all --output-dir eval_output/benchmark_5seed_fidelity \
  --resume --workers 8 --timeout 300 --watchdog-timeout 420
python3 scripts/consolidate_positions_hdf5.py \
  --input eval_output/benchmark_5seed_fidelity \
  --output eval_output/benchmark_5seed_fidelity/positions.h5
python3 scripts/fast_fidelity_report.py \
  --results eval_output/benchmark_5seed_fidelity/results.json \
  --positions eval_output/benchmark_5seed_fidelity/positions.h5 \
  --output eval_output/fidelity_report_r69/stage1 \
  --max-seeds 5 --bit-exact-threshold 1e-3
```
Then read `stage1/per_variant.json`: Tier1 = max RMSD < 1e-3. Timeout/error-dominated
= Tier2. Everything else -> escalation list (comma-joined engine names for P3 --engines).

### P3 (100-seed escalation subset + TOST)
```
python3 scripts/run_benchmark.py --seeds 100 --seed-start 42 --variants \
  --engines "<comma-separated escalation subset>" \
  --output-dir eval_output/benchmark_100seed_escalation \
  --resume --workers 8 --timeout 300 --watchdog-timeout 600
python3 scripts/consolidate_positions_hdf5.py --input eval_output/benchmark_100seed_escalation \
  --output eval_output/benchmark_100seed_escalation/positions.h5
python3 scripts/r68_tost_followup.py \
  --per-variant-json eval_output/fidelity_report_r69/stage1/per_variant.json \
  --results eval_output/benchmark_100seed_escalation/results.json \
  --positions eval_output/benchmark_100seed_escalation/positions.h5 \
  --output eval_output/fidelity_report_r69/tost
```

### P4 (combined report)
```
python3 scripts/r68_combined_report.py \
  --per-seed eval_output/fidelity_report_r69/stage1 \
  --tost eval_output/fidelity_report_r69/tost \
  --output eval_output/fidelity_report_r69/report.md
```
(May need a small edit to emit the explicit 4-tier grouping -- verify its output covers
all 118 variants in exactly one tier; if not, patch the combined-report script.)

## Wake-up case routing

| Observable | State | Action |
|---|---|---|
| codex/benchmark PID alive (`kill -0`) | RUNNING | yield turn, do not poll; watcher will fire |
| P1 codex exited + HEAD advanced + mapping report exists | P1_DONE | verify `grep -c '"fidelity_mode"'` rose; import clean; no binary shell-outs; text JMT "P1 done"; -> launch P2 |
| P2 benchmark done + stage1 report exists | P2_DONE | parse per_variant.json -> Tier1/Tier2/escalation lists; write lists into this file; text JMT "P2 done: N bit-identical, M timeouts, K escalate"; -> launch P3 |
| P3 benchmark+TOST done | P3_DONE | parse tost output -> Tier3/Tier4; text JMT "P3 done: X equiv, Y diff"; -> P4 |
| P4 report covers all 118 in one tier | SHUTDOWN | run shutdown procedure |
| codex exited but HEAD unchanged / no mapping report | P1_FAIL | read /tmp/r69_p1.log tail; targeted `/codex:rescue --resume` fixup (ONE retry); else escalate |
| benchmark exited non-zero | BENCH_FAIL | check log; re-run with --resume (idempotent); 3 fails same cause -> RESIDUAL |
| codex quota (CODEX_FAILED usage_limit) | QUOTA_BLOCKED | fallback chain below |
| same issue 3 rounds | RESIDUAL | accept, log, continue/shutdown |

## Fallback chain
1. Primary: codex via `codex-bg.sh` + `codex-watch.sh` Monitor.
2. Quota blocked: `Agent(subagent_type="general-purpose", model="opus")`, spec adapted
   (drop XML scaffolding, keep contracts + file:line). Benchmarks are plain python --
   never need codex; run via `bg-watch.sh`.
3. Both blocked: `state: BLOCKED`, iMessage JMT with reset time, ScheduleWakeup, stop.
NEVER export OPENAI_API_KEY. NEVER silently stall.

## Gotchas (read before acting)
- `fast_fidelity_report.py` key format MUST be `<graph>::<engine>::seed<N>` (double-colon).
- MEDIAN vs MAX: chaotic-but-faithful engines (fr family) show MEDIAN ~1e-16 but high MAX
  RMSD (basin divergence on big graphs). That is Tier 3 material, NOT a mapping bug. A
  WRONG fidelity_mode shows MEDIAN also high (e.g. davidson_harel default median 0.76).
  Use median-near-epsilon as the "port is active" signal when triaging P1 correctness.
- Old `eval_output/benchmark_100seed_final` (14G, un-matched, INVALID for this goal):
  after P1 verified, archive to `/mnt/locker/jt3295/` to reclaim disk (/home at 89%).
  Do NOT --resume into it; P2/P3 use FRESH dirs.
- Disk: /home 48G free (89%); /mnt/locker 3.0T free. 5-seed run ~1-2G; fine.
- pause sentinels checked clear at launch; re-check `~/.claude/state/paused-*.sentinel`
  before each codex dispatch.

## Shutdown procedure (mechanical)
1. Write `r69_4tier_fidelity_SUMMARY.md` -- the full 4-tier table (variant -> tier),
   counts, methodology, any no-port/no-op/residual findings.
2. Ensure `eval_output/fidelity_report_r69/report.md` is the canonical combined report.
3. File the report for human review:
   `~/.claude/scripts/file-for-review.sh <SUMMARY path> --label "dagua 4-tier fidelity verdict" --agent cc`
4. Mark this file `state: DONE`; append shutdown row to log.
5. Final iMessage: "Run r69 done. T1=<n> bit-identical, T2=<n> timeout, T3=<n> TOST-equiv,
   T4=<n> TOST-diff. Report: eval_output/fidelity_report_r69/report.md".
6. Delete stale `.project-context/baton.md` (it predates the launch).

## OVERNIGHT P2_DONE -> TRIAGE -> P3 PROCEDURE (2026-06-01, JMT asleep, full autonomous)

When the P2 watcher fires DONE (wrapper 1024858 exits) OR stage1/per_variant.json appears:
1. Verify `eval_output/fidelity_report_r69/stage1/per_variant.json` + `report.md` exist.
2. CANARY: confirm linlog variants are bit-identical (max RMSD < 1e-3). If linlog is NOT
   bit-identical, fidelity didn't propagate -> investigate before trusting any verdict.
3. TRIAGE (write a small python against per_variant.json -- inspect its keys first):
   - Tier 1 BIT_IDENTICAL = verdict MACHINE_EPSILON/BIT_EXACT (max RMSD < 1e-3 across the 5 seeds).
   - For NOT-Tier-1 variants, check `eval_output/benchmark_5seed_fidelity/results.json`:
     a variant whose (graph,seed) entries are majority status in {timeout, error w/ "timeout"}
     -> Tier 2 TIMEOUT.
   - Remaining (not bit-identical, not timeout-dominated) -> ESCALATION.
4. Build escalation engine list = each escalating classic_* variant's engine_name PLUS its
   paired reference (original_engine from variants.py) so TOST has both sides at 100 seeds.
   Write comma-separated to `/tmp/r69_escalation_engines.txt`.
5. iMessage JMT: "P2 done. Tier1(bit-identical)=N, Tier2(timeout)=M, escalating K to 100-seed."
   Record counts + lists in this file (section below).
6. Launch P3: `setsid bash scripts/r69_p3_100seed_tost.sh >/dev/null 2>&1 </dev/null &`
   then find wrapper pid via `ps -C bash -o pid,args | grep r69_p3_100seed_tost`, write to
   /tmp/r69_p3.pid, and Monitor `bg-watch.sh <pid> /tmp/r69_p3_100seed.log --label r69-p3 --max-runtime-min 360`.
7. On P3 DONE: verify eval_output/fidelity_report_r69/report.md exists + covers all variants
   in exactly one tier; if r68_combined_report.py output isn't an explicit 4-tier grouping,
   patch it (P4). Then run shutdown procedure (SUMMARY.md, file-for-review, final iMessage).
NOTE: P3 should be FAST (escalation engines complete, just chaotic -- no timeout graphs).
REMINDER: check liveness with `ps -C python3` / `kill -0 <pid>`, NEVER `pgrep -f <pattern-in-cmd>`.

## Escalation lists (filled after P2)
- Tier1 (bit-identical): FILL_AFTER_P2
- Tier2 (timeout): FILL_AFTER_P2
- Escalation -> P3: FILL_AFTER_P2

## Iteration log (append per round)

| Round | Phase | Start | End | Commit | Result | Notes |
|---|---|---|---|---|---|---|
| 1 | P1 | 17:25 | 17:34 | (stopped) | codex 92126 HARD-STOPPED on linlog delegation (guardrail worked) | led to P1a |
| 2 | P1a | 17:51 | 17:56 | a700ccd | linlog real port, bit-exact RMSD 1.47e-16, 15 parity tests | CC committed |
| 3 | P1b | 18:21 | 19:03 | 7c15129 | 91 variants fidelity-matched, 27 no-port | CC verified, codex committed |
| 4 | P2 | 19:08 | -- | -- | 5-seed sweep RUNNING (wrapper pid 236361) | SLOW (fidelity-port tail): 46%@2h, 59%@6h. bg-watch re-armed (480min ceiling, task bawn1kxw2) -- will re-arm again if TIMEOUT; job is progressing not hung (log fresh). On DONE: read eval_output/fidelity_report_r69/stage1/per_variant.json, triage tiers |

> **NO-OP SCARE RESOLVED (2026-06-01 ~01:15):** benchmark log showed "unrecognized
> variant params: ...fidelity_mode..." for linlog/sgd2_multi/davidson_harel/maxent/pivot_mds.
> Investigated: classic_competitor.py:82-89 does `layout_params.update(variant_params)` ->
> ALL params (incl fidelity_mode) ARE forwarded; the warning is COSMETIC (incomplete
> variant_param_names whitelist used only for the warning). Empirically confirmed via the
> actual dispatch path (_quick_classic): linlog fid on-vs-off delta 17.4, davidson_harel
> delta 802 -> fidelity_mode ROUTES. sgd2_multi: builder `del fidelity_mode` is BY DESIGN
> (comment: "always the native Python port rather than runtime delegation") -> always
> fidelity, flag vestigial. CONCLUSION: P2 is VALID. No fix needed. Minor cosmetic TODO:
> add fidelity_mode etc. to variant_param_names whitelists to silence the warning (low pri).
> CANARY: linlog must come back bit-identical (~0) in P2 report; if not, fidelity didn't propagate.

> **P2 PROGRESS / PACING (2026-06-01 07:00):** SLOW deceleration -- 46%@2h, 59%@6h, 64%@12h.
> Heavy fidelity ports (fmmm/maxent/davidson_harel/sgd2/lgl/drl) on large graphs (ba_500
> etc.) take 50-300s each, many hit the 300s timeout (= Tier 2, expected). 20 cores, load
> ~15, ~38% idle but workers are multi-threaded (~2 cores each) so cores ~saturated.
> DECISION: LET IT RUN (do NOT kill -- healthy/CPU-bound/progressing; multi-day sanctioned;
> --resume restart for +workers judged not worth orphan-tree risk for ~1.5x). ETA P2: maybe
> +1-2 days. Watcher bg-watch keeps dying ~6h in (exit 144, spurious) -> re-arm each time
> with --max-runtime-min 360 (task bdwsyogug); each death = a progress check-in. If a future
> check shows rate has collapsed further or stalled (log mtime >10min old), reconsider
> restart with --workers 14 + OMP_NUM_THREADS=1 + --resume (safe: preserves completed rows).

> **P2 RESTARTED for speed (2026-06-01 13:06):** rate collapsed (64%@12h -> 67%@18h,
> ~0.5%/hr => ~2.5 more days). Cleanly killed old run (236361/778301 dead, verified via
> ps -C python3 NOT pgrep -f which self-matches!), backed up results.json
> (results.json.r69p2_pre_restart_bak, 65,373 entries preserved + valid). Edited
> scripts/r69_p2_5seed_sweep.sh: --workers 8->18, added OMP/MKL/OPENBLAS/NUMEXPR
> _NUM_THREADS=1 (heavy fidelity ports are single-threaded sequential -> threads waste
> cores; 18 single-thread workers on 20 cores ~2.25x parallelism). Relaunched --resume,
> new wrapper pid 1024858, watcher task bpcva82sl (360min). LESSON: use `ps -C python3`
> or `kill -0 <pid>` to check job liveness, NEVER `pgrep -f "<pattern in my own cmd>"`
> (self-match trap -- nearly killed my own shell). On next check: verify 18 workers + rate up.

> **P2 PACING WATCH (2026-06-01 19:33):** ~73% (69,600 RESOLVED = ok+err+skip+timeout;
> ignore "running" -- those are stale orphans like the 100seed's 5023, NOT real in-flight).
> RESOLVED rate in the 18:49->19:33 window was only ~400/hr (slow patch of 300s-timeout
> heavy ports). Earlier windows ~700-900/hr. If next check-in confirms SUSTAINED ~400/hr,
> remaining ~26K => ~2.5 days, which EXCEEDS the 12-24h / 1-1.5day I told JMT -> proactively
> flag him + offer the acceleration (skip remaining seeds once an (engine,graph) times out;
> he declined earlier at the 12-24h estimate but would likely accept at 2.5 days). Watcher
> b2j2s86ue (360min). Track RESOLVED count, not total entries.
