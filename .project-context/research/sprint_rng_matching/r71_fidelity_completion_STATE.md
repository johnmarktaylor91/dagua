---
run: r71_fidelity_completion
created: 2026-06-12
state: PHASE_PLAN_REVIEW
current_round: 0
max_adversarial_rounds: 3
gate_file: .project-context/autonomous_gate_r71.json
plan: PLAN_r71_fidelity_completion.md
---

# r71 -- Fidelity Completion (autonomous run)

JMT directives (2026-06-12, verbatim): "im fine calling it 1b, they deserve to be marked
as basically faithful" (DONE e9e91c7) / "can u verify this? if not, plz kick off running
as many seeds as you need" (VERIFIED: graphviz/ogdf/igraph-det refs never seeded anywhere
-> P1) / "plz make a plan for investigating and fixing the failures, do up to three rounds
of adversarial fable critique, then execute the plan. the goal is 100% fidelity." /
"bigger compute budget is fine. and yes do the sugiyama thing" (W4 RUNNING pid 531430
watcher b40wvbbg6; W5 queued) / "feel free to go autonomous until end of the pipeline".

## Phase ladder
| Phase | What | Done when |
|---|---|---|
| R | Fable plan review <=3 rounds | PASS or 3 rounds (residuals documented) |
| W4/W5 | toolkit 1200s rerun; sugiyama rung-0 reclass | det verdicts updated; 5 combos reclassified |
| P1 | seeded-reference upgrade (audit -> adapters -> probe -> bench -> re-analysis) | upgraded combos have Mode A verdicts; unseedable documented |
| P2 | Tier-4 root-cause loop (cluster -> diagnose -> fix -> re-verify) | every surviving Tier-4 combo root-cause-labeled |
| P3 | data-gap repair (timeouts, missing refs, STRUCTURAL_NA) | zero unexplained INSUFFICIENT; coverage floors met |
| P4 | report v2 + archive r70 + supersession + text JMT | gate all-pass; state DONE |

## Wake-up routing
| Observable | Action |
|---|---|
| Fable FAIL + findings | revise plan, round+1 (<=3), redispatch |
| Fable PASS | execute: P1a codex dispatch + P2a cluster table (parallel) |
| W4 watcher fires | count DIFFERENT->INVARIANCE_EQUIVALENT flips; record; W5 next |
| codex CODEX_DONE | verify per prompt contract; commit; next |
| benchmark watcher fires | verify counts; next phase step |
| quota / 3-fail / wedge | global fallback chain; BLISS pattern for any toolkit call |

## Key facts (verified)
- Old benchmark_100seed_final (709,725 rows): seeded refs ONLY for Python/igraph-stochastic
  families (sgd2, umap, davidson, drl, graphopt, lgl -- already Mode A). graphviz/ogdf/
  igraph-det refs: single deterministic rows in every store. P1 compute is genuinely new.
- W4 budget env: DAGUA_R70_TOOLKIT_BUDGET_S (b2dbced); resume dedups (rewrites kept rows).
- Tier 1b live in generator (e9e91c7): TIER_1B_INVARIANCE_EXACT headline at >=90%
  invariance-equivalent.
- r70 canonical numbers: 3,955 combos; rung1 883 / rung2 299 / rung2' 1,503 / rung3 318 /
  rung4 705 / INSUF 247; det 721/71/40/8; 22 UNDETERMINED engines.

## Iteration log
| Round | Phase | When | Result |
|---|---|---|---|
| 0 | setup | 2026-06-12 | Tier-1b shipped; seeded-originals verified ABSENT; W4 launched; plan v1 written |
| 1 | R | 2026-06-12 | Fable round 1: FAIL, 11 findings (5 HIGH). HEADLINE: seed plumbing ALREADY EXISTS (graphviz -Gseed r9, ogdf_runner setSeed r28, live-probed: sfdp/neato/fdp/gem/fmmm/ogdf_stress VARY); blocker is _BASE_ENGINE_STOCHASTICITY -> [None] seeds; igraph_sugiyama drops seed (adapter bug). P1 collapses to --seed-refs override + igraph fix; P1d cost ~10-13 CPU-h. Also: CHAOTIC_BASIN redefined ensemble-level; P2 Mode-B gated on P1e; fixed-engine full re-runs; disk pre-step (35G < 10% floor). ALL accepted -> plan v2 (Appendix A). Reviewer id acab06233580cf58c. |
| 2 | R | 2026-06-12 | Fable round 2: FAIL, 7 findings (2 HIGH: eps-init CHAOTIC_BASIN control unexecutable for binary refs -> seed-split self-ensemble; data-SHA claim machine-uncheckable -> stamp+source_dir+assertion). BONUS: Mode-A Tier-4s concentrate on weighted/multiedge graphs (codeable bug, not chaos); sgd2 = TRACKING_BUT_SHIFTED triage-first. ALL accepted -> plan v3 (Appendix B). Reviewer id a62c39f1a1386f3a2. Round 3 = final. |
| 3 | R | 2026-06-12 | Fable round 3: **PASS** (0 HIGH; 7 wording fixes applied -> plan v4 APPROVED, committed 3e429fd). Reviewer id aa4396bfdfa0711bf. EXECUTION BEGINS: NOW block = P1c disk archive + P1a codex (i) + P2a cluster table; W4 still running (watcher b40wvbbg6). |
| 4 | NOW block | 2026-06-12 | IN FLIGHT: (a) W4 toolkit 1200s rerun pid 531430 watcher b40wvbbg6; (b) P1c disk archive (benchmark_100seed_final + benchmark_100seed_escalation -> /mnt/locker/jt3295/dagua_archives) rsync 729264, watcher b62ud6wv6, log /tmp/r71_disk_archive.log; (c) P1a(i) codex seed-refs override pid 731344 watcher bku0z84d1 prompt PROMPT_R71_P1a_i_seed_refs.md; (d) P2c umap-weights codex pid 740171 watcher bcehgdim0 prompt PROMPT_R71_P2c_umap_weights.md (parallel-safe: disjoint files). P2a cluster table DONE -> eval_output/fidelity_definitive/r71_cluster_table.json (705 T4 -> 27 clusters; Mode-A: umap 18 plain e_rel .05 + 16 weighted e_rel .62 disp .62; drl 15 weighted; sgd2 14 weighted 11xTBS). Weight-hypothesis evidence: pipelines/umap.py has ZERO weight handling, ref adapter has 3 weight refs. ON P1a DONE: verify+commit -> dispatch P1a(ii) (provenance). ON P2c DONE: review diagnosis, commit fix, queue umap failing-map re-run. ON archive DONE: verify df>=50G. ON W4 DONE: count flips, W5 next. |
