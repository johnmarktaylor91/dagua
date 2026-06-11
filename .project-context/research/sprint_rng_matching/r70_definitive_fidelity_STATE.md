---
run: r70_definitive_fidelity
created: 2026-06-11
state: PHASE_IA_CODEX_STATS_CORE
current_phase: I-A
max_adversarial_rounds: 5
gate_file: .project-context/autonomous_gate_r70.json
---

# r70 -- Definitive Distributional Fidelity Analysis (autonomous run)

JMT directive (2026-06-11, verbatim): "plz write a spec, do at most five rounds
adversarial review with a fable or until it passes, dispatch codexes to code it up,
then kick off the analysis. do it all autonomously plz and text me when done ...
we should aim to make this our definitive fidelity analysis, incorporating everything
we might want to know. its a good time to think CAREFULLY about this stuff. and we
should consider it as superseding previous analyses. make sure to clearly document
everything you did ... q95 is fine."

Spec: ./SPEC_definitive_fidelity_analysis.md (the single source of methodology truth).
Supersedes: WHERE_WE_STAND.md group tables, ALLGRAPHS_SUMMARY.md, fidelity_report_r69,
fidelity_report_final verdicts (the 5-seed triage REMAINS the input for tier-1/escalation
scoping; its per-variant PARTIAL verdicts are superseded by r70 verdicts).

## Phase ladder

| Phase | What | Done when |
|---|---|---|
| S  | Write SPEC_definitive_fidelity_analysis.md | spec file exists |
| R  | Adversarial review, Fable agent, <=5 rounds | reviewer verdict PASS, or round 5 done (residuals documented in spec appendix) |
| I-A | Codex: stats core dagua/eval/distributional_fidelity.py + tests | commit; pytest green; complex-Gram == procrustes_rmsd agreement test passes |
| I-B | Codex: runner scripts/definitive_fidelity_analysis.py | commit; smoke on ~20 combos produces sane per_combo.jsonl |
| I-C | Codex: report scripts/definitive_fidelity_report.py (aggregation, four-tier, deterministic integration) | commit; report renders on smoke data |
| CB | Control benchmarks (CC writes small scripts, runs via bg-watch): tier1 positive-control 100-seed + deterministic-refresh 5-seed | both benchmark dirs complete |
| V  | CONTROLS GATE: positive / negative / chance criteria (spec sec. 8) | all three pass. FAIL -> debug harness, do NOT run full pass |
| F  | Full run on all 3,955 escalation combos + deterministic verdicts | per_combo.jsonl covers 100% of combos (verdict or INSUFFICIENT_DATA) |
| G  | DEFINITIVE_FIDELITY_REPORT.md + FOUR_TIER_CATEGORIZATION.md, supersession notes, commit, file-for-review, text JMT | gate file all-pass; state DONE |

## Wake-up case routing (read FIRST every wake-up)

| Observable | Action |
|---|---|
| Fable Agent returned FAIL + findings | revise spec, increment round, redispatch reviewer (<=5) |
| Fable Agent returned PASS | -> I-A: dispatch codex A (codex-bg.sh + codex-watch.sh Monitor) |
| codex CODEX_DONE | verify (git log, pytest, files exist); commit if codex didn't; -> next codex / phase |
| codex CODEX_FAILED quota | fallback chain below |
| codex CODEX_FAILED other | read log tail; ONE targeted retry; else hand-implement if small, else escalate |
| control benchmark DONE | run controls gate analysis (phase V) |
| controls gate FAIL | STOP full pass; diagnose harness (historic pattern: ~1/3 of failures are harness bugs); fix; re-run controls; 3 fails same cause -> text JMT BLOCKED |
| full-run watcher DONE | verify coverage; -> phase G |
| 3 rounds same un-closeable issue | accept residual, document, continue |

## Fallback chain
1. codex via codex-bg.sh (effort medium) + codex-watch.sh Monitor.
2. Codex quota-blocked -> Agent subagent (sonnet default; this is mechanical impl). Benchmarks +
   analysis runs are plain python -- never need codex; run via bg-watch.sh.
3. Both blocked -> state BLOCKED, text JMT reset time, ScheduleWakeup.
NEVER export OPENAI_API_KEY. Check pause sentinels before each codex dispatch.
Liveness checks: kill -0 PID / ps -C python3. NEVER pgrep -f (self-match).

## Data inputs (verified 2026-06-11 against the actual files)
- eval_output/benchmark_100seed_escalation_final/: results.json (541,124 rows, dict keyed
  `graph::engine::seedN`), positions/*.pt (514,735 files, torch [N,2]). Seeds 42-141 BOTH sides.
  128 engines (64 reimpl + 64 refs named `<ref>__for__<classic_variant>`). umap merged in.
- .project-context/research/sprint_rng_matching/failing_map_final.json: {engine:{ref,graphs}},
  64 engines, 3,955 combos.
- eval_output/benchmark_5seed_final/ (+ positions.h5) and eval_output/fidelity_report_final/
  {per_variant.json (summary + failures[variant]=[graph,seed,rmsd]), triage_final.md}:
  39 BIT_IDENTICAL / 64 ESCALATE / 8 DETERMINISTIC_DIFFERENT / 0 TIMEOUT / 4 NO_REFERENCE /
  3 UNVERDICTED_OTHER = 118 variants.
- dagua/eval/graph_generator.py: graph registry with tags (directedness/hierarchy for domain map).
- dagua/eval/equivalence_metrics.py: 5-invariance toolkit (for deterministic engines + escalation
  re-scoring of borderline combos).
- scripts/fast_fidelity_report.py:26 procrustes_rmsd = the reference distance convention.
- CAVEAT (from sprint STATE 2026-06-03): benchmark_5seed_final sugiyama positions are STALE
  pre-closing-wave -> deterministic-refresh benchmark re-runs all 8 deterministic engines fresh.

## Resource notes
- Disk 36G free (92%) -- analysis outputs are small (JSON/MD); control benchmarks ~1-2G. OK.
- Compute: complex-Gram Procrustes makes per-combo stats sub-second; full pass est. <=1-2h
  on 12 workers; I/O (~750k torch.load) dominates.

## Shutdown procedure (mechanical)
1. Verify gate file: every criterion status=pass.
2. DEFINITIVE_FIDELITY_REPORT.md + FOUR_TIER_CATEGORIZATION.md in eval_output/fidelity_definitive/.
3. Supersession: update WHERE_WE_STAND.md header (point to definitive report), add supersession
   notes to ALLGRAPHS_SUMMARY.md + fidelity_report_r69/triage.md headers. Delete stale
   .project-context/baton.md (pre-R69, long superseded).
4. Commit (conventional, NO AI attribution). file-for-review.sh the definitive report.
5. Text JMT: headline (N engines distributionally matched / seed-faithful, tier counts), report path.
6. state: DONE.

## Iteration log

| Round | Phase | When | Result |
|---|---|---|---|
| 0 | S | 2026-06-11 | recon done (schemas verified); state+gate+spec written |
| 1 | R | 2026-06-11 | Fable round 1: FAIL, 17 findings (2 CRIT: Mode-B deterministic refs = 62% of combos; near-det guard below float32 floor). ALL accepted -> spec v2 (Appendix A). Reviewer agent id a4cebf0e980beaefa. |
| 2 | R | 2026-06-11 | Fable round 2: FAIL, 14 findings (2 CRIT: conformal-p floor x BH cliff; Mode B missing point-mass guard. 4 HIGH: gem zero-power typicality, dead timeout recipe, stale sgd2 claim, negctl flake). ALL accepted -> spec v3 (Appendix B). Reviewer id a0920cd3e95015b5d. |
| 3 | R | 2026-06-11 | Fable round 3: FAIL, 12 findings (2 CRIT: "usable"/headline scope undefined; accounting partition non-exhaustive (169 fall-through combos). 3 HIGH: gate-2 vs informativeness guard tension, asymmetric anisotropic distance, REF_COMPATIBLE informative floor). Per-combo traces CLEAN. ALL accepted -> spec v4 (Appendix C). Reviewer id a907072e0f0aa4a83. |
| 4 | R | 2026-06-11 | Fable round 4: FAIL but narrow -- 2 HIGH (ogdf_stress shared-ref collision in negctl; headlines undefined for 39 bit-identical + 8 deterministic engines), 6 MED/LOW post-PASS nits. ALL 8 fixed -> spec v5 (Appendix D). Reviewer id a5c8dce8de63d3cc2. Round 5 = final allowed; scope narrow per reviewer. |
| 5 | R | 2026-06-11 | Fable round 5: **PASS** (0 HIGH/CRIT; 6 MED/LOW clarifications, all applied inline -> spec v6 APPROVED). Reviewer id a1e624f31003d95ac. Gate criterion 1 PASS. -> Phase I-A: codex Task A (stats core), prompt at PROMPT_R70_taskA.md. |
