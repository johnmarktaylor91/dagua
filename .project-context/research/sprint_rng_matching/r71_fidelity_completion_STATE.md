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
| 5 | P1a-i/P1b done | 2026-06-12 | P1a(i) VERIFIED+committed 38a1bc4 (--seed-refs both call sites, igraph_sugiyama seed fix, 5 tests). P1a(ii) provenance codex RUNNING pid 757606 watcher bytb7r0jq. **P1b PROBE COMPLETE** (positive control GREEN; committed bc5847b; table eval_output/fidelity_definitive/r71_seedability_probe.json with upstream evidence): SEEDABLE = neato, sfdp, ogdf_fmmm, ogdf_gem, ogdf_stress, igraph_mds (SURPRISE: classical_mds upgrades! cross 1.3); fdp = UNSTABLE_WITHIN_SEED_ENSEMBLE_OK (dist test valid, tracking N/A); DETERMINISTIC-with-evidence = graphviz_dot, ogdf_pivot_mds, igraph_sugiyama (flat 5e-20 even post-fix). P1d launcher READY scripts/r71_p1d_seeded_refs.sh -- **LAUNCH ONLY AFTER P1a(ii) CODEX_DONE** (it edits run_benchmark.py; avoid import race). ON P1a(ii) DONE: verify, commit, smoke 1 engine x 2 seeds, then launch P1d via setsid + correct-PID bg-watch. |
| 6 | P1d LAUNCHED | 2026-06-12 | P1a(ii) VERIFIED+committed 7b909a0 (git_sha in rows+manifest; source_dir merge tags; fixed_engines.json report assertion -- smoke: synthetic config FAILS strict, absent config passes). Seeded-path smoke GREEN (sfdp synthetic ref -> ::seed42/::seed43 keys, cross-seed 0.074). **P1D RUNNING**: pid 813746, watcher biyanbvbb, log /tmp/r71_p1d.log, 24 ref engines x failing-map graphs x seeds 42-141 -> eval_output/benchmark_100seed_seeded_refs (disk floor guard 15G; disk 42G free, archive still moving). ON P1D DONE: merge into union store (CC = serialized merge owner; merge_benchmark_datasets with source_dir tags) -> P1e re-analysis of upgraded combos -> re-cluster -> Mode-B fix loop. Still in flight: W4 toolkit (b40wvbbg6), disk archive (b62ud6wv6), P2c umap codex (bcehgdim0). |
| 7 | P2c umap FIXED | 2026-06-12 | umap diagnosis: weight ROUTING existed (ops/umap.py, not pipeline file); root cause = preprocessing semantics mismatch vs adapter (distance/cost + duplicate-edge coalescing). Fix committed 0766227 (+tests, 25 green). Matched-seed before/after: weighted_clusters 0.067->0.000000 BIT-EXACT; heavy_tail 0.105->0.058; plain control unchanged. fixed_engines.json registered (6 umap variants, pre_fix_dirs = escalation_final + umap_rerun). UMAP POST-FIX RE-RUN launched: pid 831545, watcher b47tv2aiy, log /tmp/r71_umap_rerun.log -> eval_output/benchmark_100seed_umap_postfix (full failing map, 100 seeds, 4 workers alongside P1d's 10). NOTE: check_engine.py hung on umap (no fixtures) -- codex used direct 5-seed fallback; fine. NEXT fix targets after P1e: drl weighted (15), sgd2 TBS triage (18). |
| 8 | LOAD REDUCED (JMT request) | 2026-06-12 | JMT wants the machine for other work; analysis may chug into the weekend. P1d + umap reruns STOPPED cleanly and RELAUNCHED resume-safe at nice 19 / ionice idle with reduced workers (P1d 10->4 pid 1110197 watcher buqtkqtix; umap 4->2 pid 1110196 watcher be5w2cx0q; watcher ceilings 2880min). W4 + workers reniced 19. Total analysis cores ~10 at IDLE priority -- JMT's interactive work preempts everything. ETAs stretch ~2-3x: P1d ~10-16h, umap ~3-6h, W4 ~4-8h. Routing unchanged: ON P1D2 DONE -> merge -> P1e. |
