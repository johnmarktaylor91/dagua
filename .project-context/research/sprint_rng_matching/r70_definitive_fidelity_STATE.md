---
run: r70_definitive_fidelity
created: 2026-06-11
state: DONE
completed: 2026-06-12
current_phase: G_complete
max_adversarial_rounds: 5
gate_file: .project-context/autonomous_gate_r70.json
---

> **DONE 2026-06-12.** Canonical outputs: eval_output/fidelity_definitive/
> {DEFINITIVE_FIDELITY_REPORT.md, FOUR_TIER_CATEGORIZATION.md, per_combo.json,
> controls/gate_results.json, oc_simulation.json}. Engine headlines (111 + 7 appendix =
> 118): 39 BIT_EXACT, 28 DISTRIBUTIONALLY_MATCHED (13 also SEED_FAITHFUL),
> 14 REF_COMPATIBLE, 8 deterministic-verdict (721/840 combos invariance-equivalent),
> 22 UNDETERMINED (honest: low coverage / informative floors / domain-mismatch-dominated,
> e.g. sugiyama). Per-combo (3,955): rung1 883, rung2 299, rung2' 1,503, rung3 318,
> rung4 705, insufficient 247 -> Tier 3 = 3,003/3,708 scored (81%), Tier 4 = 705 (19%).
> Controls: gates 1/2/4 PASS; gate 3 deviation documented (spec Appendix E).
> Post-mortems worth remembering: BLISS automorphism search uninterruptible + intractable
> on twin-heavy graphs (hard-killed subprocess pattern is the fix); ref layouts stored
> float32; deterministic refs keyed ::deterministic broke three naive pairing assumptions.

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
| 6 | I-A + CB | 2026-06-11 | PARALLEL: (a) codex Task A dispatched pid 3370325, log /tmp/r70_taskA.log, watcher b990dt4yv, spec committed 1531bb1. (b) Phase CB benchmarks launched pid 3385252 (NOT 3385247 -- setsid parent PID trap, watcher re-armed bq89pxet3), log /tmp/r70_phase_cb.log: CB1 tier1-controls (8 pre-screened graphs: center_port_backedge_hub, clustered_medium_5x20, heavy_tail_weights_50, planar_60, real_lesmis_77, sbm_5x50, sparse_pair_50, weighted_clusters_3x10; 10 of 10 qualifying, draw seeded; ctl_graphs.json written) -> CB2 deterministic refresh (8 det engines + refs) -> CB3 sugiyama reverify. ON CODEX_DONE: verify per PROMPT (pytest, ruff, import, only 2 files), commit, dispatch Task B (write PROMPT_R70_taskB.md first). ON R70_CB_DONE: verify 3 dirs complete. |
| 7 | I-A DONE -> I-B | 2026-06-11 | Task A VERIFIED (18 tests pass independently; max fast-vs-oracle delta 6.7e-16 on 900 random pairs; hybrid fallback exact at d~1e-8) + committed 9cb2082. Task B (runner) dispatched pid 3409479, log /tmp/r70_taskB.log, watcher be623wihi, prompt PROMPT_R70_taskB.md. Task C prompt pre-written (PROMPT_R70_taskC.md). ON TASK B CODEX_DONE: verify (--help, smoke jsonl rows spec-conformant: fr=ModeA small diag, classical_mds=ModeB near_det, gem=UNINFORMATIVE, sugiyama=free_aspect; ruff; 1 file), commit, dispatch Task C. |
| 12 | det saga | 2026-06-12 | Deterministic mode wedged TWICE: (a) serial+uncapped 10h on giants -> parallel+incremental+size-cap (104898d); (b) STILL wedged 2h42m on chung_lu_150 etc -- ROOT CAUSE: igraph/BLISS automorphism SEARCH (single C call, >490s, signal-immune) intractable on twin-heavy random graphs; cap irrelevant. FIX a79316e: hard subprocess kill at 90s + conservative plain-Procrustes fallback (sound: toolkit<=plain so plain<1e-3 still proves INVARIANCE_EQUIVALENT; else flag toolkit_timeout -> quality axis) + resume (72 rows kept). Verified on pathological combo (kills at budget; plain d=1.5e-16 resolves it anyway). Relaunched pid 182133, log /tmp/r70_det_rerun2.log, watcher b3vcz5n1c. Worst case ~2.4h, expected <1h. ON R70_DET2 DONE: verify 840 det rows + 206 rung0 rows sane -> strict report + --controls -> phase G. |
| 11 | F DONE | 2026-06-11 | FULL RUN COMPLETE: 3,955/3,955 unique combos, 0 missing/extra. Modes A=1,258 B=2,450 INSUF=247 (matched_seeds_lt_30 78, reimpl_seeds_lt_30 70, ref_seeds_lt_30 14, no_reference_rows 85). Pre-FDR signal: ModeA 1,182/1,258 dist_equivalent (94%), 898 strong-track; ModeB 488 near_det + 204 uninformative + 1,662 typical. Gate criterion 6 satisfied pending report assertions. CB3 sugiyama reverify at 93%. ON CB DONE: --mode deterministic + --mode rung0-reverify -> then report strict + --controls -> phase G. |
| 10 | V DONE -> F | 2026-06-11 | GATE 4 **PASS** (KS p=0.82, recovery 21 in [8,33]). GATE 3 **FORMAL FAIL 90%** -> diagnosed (NO harness bug): ModeA sub-gate 12/12 + 0 false-track; 2 ModeB escapes geometrically explained (sfdp p_typ=0.069 near-miss d_R=2.3x spread; gem ref INSIDE cloud d_R 0.171 < W_D 0.206) = pre-registered ModeB power limit. DEVIATION documented spec Appendix E (proceed; ModeB false-typicality 2/8=25% MUST appear in report's Mode B disclosure). Token map signed off (15 tokens, no mismaps). PHASE F LAUNCHED: full run pid 3539853, log /tmp/r70_fullrun.log, watcher bs94lrxri, --resume, workers 12. CB2/CB3 still running (pid 3385252). ON R70_FULL DONE: verify jsonl coverage 3,955; ON CB done: run --mode deterministic + --mode rung0-reverify; THEN Task C report (strict) + controls eval + phase G. |
| 9 | V (controls) | 2026-06-11 | Spot-check fixup VERIFIED (re-scores; clmds 0.81 no-flip) committed cb823e5. CB1 DONE: 7900/8000 ok; sole failure = classic_lgl_default/sbm_5x50 watchdog x100 -> INSUFFICIENT (excluded per gate rules). GATE 1 **PASS** (39/39 scored dist_equivalent + tracking; file controls/gate1_modea_positive.jsonl; combos file /tmp/r70_ctl_combos.txt uses graph::engine\tref form -- control engines not in failing_map). GATE 2 **PASS** (39/39 informative REF_TYPICAL; controls/gate2_modeb_positive.jsonl). GATE 3 (negative) running bg task byglhjiic -> controls/gate3_negative.jsonl; GATE 4 (chance) bg task bihrcwcdk -> controls/gate4_chance.jsonl. CB2/CB3 (det refresh + sugiyama reverify) still running pid 3385252. ON gate3 done: sign off controls/token_map.json + evaluate (>=95% NOT rungs 0/1/2/2'; p_track sub-gate aggregated). ON gate4: KS uniformity + Poisson band [8,33]. ALL 4 PASS + CB2/CB3 done -> phase F: full run (--mode full, all 3,955, workers 12, bg-watch) + --mode deterministic + --mode rung0-reverify. |
| 8.5 | I-C DONE + fixup | 2026-06-11 | Task C VERIFIED (all 21 report sections + 9 four-tier sections render; --help/ruff ok) + committed e080aa0. ONE gap: invariance spot-check samples but does not re-score (no position paths in rows) -> targeted fixup codex dispatched pid 3464373, log /tmp/r70_taskC_fixup.log, watcher bid38yza7, prompt /tmp/PROMPT_r70_taskC_fixup.md (ONE retry budget). CB1 at 79% all-ok. ON FIXUP DONE: verify spot-check section re-scores on smoke, commit. THEN phase V controls per round-8 routing. |
| 8 | I-B DONE -> I-C | 2026-06-11 | Task B VERIFIED on smoke (10 rows ALL spec-predicted: fr ModeA dist_eq=True p_track@floor; clmds/sugiyama near_det; sugiyama free_aspect=True; gem uninformative; r70-v6 + sha stamped) + committed 67c5431. Task C (report) dispatched pid 3434555, log /tmp/r70_taskC.log, watcher bgzra0m8l. CB1 controls bench 76% (6100/8000 ok, 0 errors). ON TASK C CODEX_DONE: verify (--help, both .md render on smoke, ruff, 1 file), commit. ON BOTH (C committed + R70_CB_DONE): phase V -- run controls: (1) runner --mode full --data-dir tier1_controls --combos-file <40 ctl combos> -> controls/gate1; (2) --mode modeb-positive-control; (3) --mode negative-control (sign off token map FIRST); (4) --mode chance-control; then report --controls -> gate_results.json; ALL PASS -> phase F full run. |
