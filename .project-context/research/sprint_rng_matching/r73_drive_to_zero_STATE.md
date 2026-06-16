# r73 -- Drive-to-Zero Fidelity Sprint -- STATE

**Started:** 2026-06-15. **Directive (JMT):** "keep pushing. leave nothing on the table unless
genuinely absurd (bottom-level FP). Find ALL remaining divergence causes we can plausibly
eliminate, and eliminate them. NOTHING is deferred. Research agents, adversarial codex iteration
to convergence, dispatch, run, summarize. Autonomous till the end."

## Entry baseline (r72 FINAL -- supersedes; see r72_RESULTS.md)
Total divergent 617/3955 = 15.6% [ModeA 331 | ModeB 286]. Inventory /tmp/r73_divergent_inventory.json.

| bucket | n | mode | hypothesis |
|---|---|---|---|
| sugiyama | 231 | B (det) | tie-breaking/ordering vs graphviz dot -- BIGGEST lever, likely tractable |
| sfdp | 184 | A | FP-floor (Lyapunov 0.8/iter verified r72) -- confirm all, find non-FP outliers |
| fmmm | 88 | A | steps10 low-iter chaos + disconnected + fdp local-nbhd 11 + steps100/200 tail |
| classical_mds | 39 | B | dagua-deterministic vs igraph-stochastic ref -- reference-choice? |
| umap | 26 | A | downstream UMAP-SGD basin |
| gem | 22 | A | chaos? or findable |
| pivot | 16 | B | deterministic pivot-MDS |
| drl/neato/maxent | 11 | A | chaotic tails |
| INSUFFICIENT | 262 | -- | sgd2 81 + drl 65 compute-frontier; davidson/fr/stress small -- recover? |

Cross-cutting: 3Q quality battery NEVER run on the 286 Mode-B combos -> extend to reclassify
quality-equal deterministic layouts out of "divergent".

## Phase ladder
| Phase | What | Done when |
|---|---|---|
| R | 6-thread parallel research salvo (Sonnet, benchmark-path diagnosis) | all 6 findings in |
| P | synthesize -> plan; adversarial Codex critique of findings; iterate | plan stable |
| I | dispatch codexes (disjoint files) per fixable mechanism; adversarial review each | each committed+verified |
| B | re-benchmark fixed engines; re-analyze union (freshest-last overlay) | data merged |
| A | final report + scorecard + supersession (r72->r73) + file-for-review + text JMT | DONE |

## Research threads (Phase R)
1. sugiyama (231) -- deterministic ordering/coordinate fidelity vs dot.
2. MDS-family Mode-B: classical_mds (39) + pivot (16) -- matchable vs different-reference.
3. Mode-A stochastic tails: gem (22) + drl (5) + neato (3) + maxent (3) -- findable RNG/init/param gap vs floor.
4. fmmm (88 div + 12 insufficient) -- per-sub-bucket fixable vs floor; fdp big-graph perf.
5. umap (26) -- SGD-basin RNG/init matchability.
6. sfdp (184) floor-confirm (non-FP outliers?) + 3Q-battery extension to Mode-B + insufficient-recovery triage (262).

## SHUTDOWN (end of sprint) -- JMT directive 2026-06-15
At sprint end (after wave 2 + Phase B rebench + final summary): MERGE develop -> main and PUSH to
remote (`git checkout main && git merge develop && git push origin main`). JMT wants progress public.
Verify clean (tests/gates) before pushing; conventional history, NO AI attribution.

## Wake-up routing
- research agent returns -> collect; all 6 in -> synthesize -> adversarial codex critique -> plan.
- codex fix DONE -> verify on BENCHMARK PATH (not direct), adversarial review, commit, re-benchmark.
- benchmark DONE -> re-analyze (overlay freshest-last) -> update scorecard.
- watchdog-killed futures leave orphan workers -> kill process GROUP (kill -9 children+main) after data flush.
- controls regen must WIPE first (producer appends). --controls BEFORE report (embeds gate).

## Iteration log
| Round | Phase | When | Result |
|---|---|---|---|
| 0 | R | 2026-06-15 | r73 scaffold; complete inventory enumerated (617 div + 262 insuff); 6-thread research salvo dispatching (Sonnet, parallel). |
| 8 | SPRINT COMPLETE | 2026-06-16 | A3 clean. DEFINITIVE: total divergent 617->574 (-43, 0 regressions); escalation-divergent (ModeA) 331->309 (raw 339 incl 30 cosmetic mds B->A relabels); 3Q 32->36. WINS 43: umap 18 (parallel-edge HIGH), pivot 16 (scale HIGH), classical_mds 5 (weight HIGH = exactly the 5 weighted graphs), fmmm 3 (MAARPacking partial), neato 1 (polyomino partial). Wrote r73_scorecard_final.json + r73_RESULTS.md (supersedes r72) + project_r73 memory. Remaining 574 = deep multi-layer ports (sugiyama position.c, fmmm rotation, neato splines) + FP floor. Adversarial critique stopped 218-combo laundering. -> commit docs, verify tests, MERGE develop->main + PUSH, summarize JMT. **SPRINT DONE.** |
| 7 | mds ref-overlay artifact, A3 fix | 2026-06-16 | A2 fixed reimpl seeds but ANOTHER overlay artifact: classical_mds ref igraph_mds IS in SEEDABLE_BASES, so old seeded_refs has 100 WEIGHTED ref seed-keys; A2 (no --seed-refs) produced unweighted ref as ::deterministic (1 key) -> overlay kept BOTH (100 weighted + 1 unweighted) = mixed ref cloud -> mds flips B->A, polluted verdict. PIVOT CLEAN (ogdf_pivot not in SEEDABLE -> single key fully replaced -> 16 wins solid). A2 clean result: total divergent 617->573 (-44), 44 wins (umap 18 SOLID, pivot 16 SOLID, classical 6 SUSPECT, neato 1, fmmm 3), 0 regressions, 3Q 32->36. PASS A3 re-running mds WITH --seed-refs igraph_mds (100-seed unweighted ref, full overlay) pid 3208565 watcher bv1dk1bnn; wiped appended r73_fixes_analysis.jsonl. ON DONE: re-analyze -> CLEAN final -> finalize -> merge main + push. LESSON: overlay rebench of a SEEDED-ref engine must re-run the ref WITH --seed-refs (same 100 keys) or old seeded ref keys persist. |
| 6 | rebench done, analysis ARTIFACT found + fixing | 2026-06-15 | Re-bench (89 graphs) done clean. Analysis v1: total divergent 617->585 (-32), 3Q 32->36 (+4), 32 wins (umap 18, pivot 9, fmmm 4, neato 1), 0 regressions. BUT ModeA 331->343 (+12) ARTIFACT: 44 classical_mds combos flipped B->A because PASS A ran mds/pivot at --seeds 5 while old data had 100 -> overlay only replaced seeds 42-46, leaving 95 stale WEIGHTED-ref seeds = Frankenstein combos (n=100 mixed ref semantics). CLEAN: umap/fmmm/neato (Pass B, full 100 seeds) wins solid (umap 18 standout = parallel-edge fix; fmmm 4 packing; neato 1; +6 INSUFFICIENT->A newly-measured neato3/fmmm3). mds/pivot CORRUPTED -> PASS A2 re-running mds+pivot at 100 seeds pid 3153033 watcher baxbew7sy (deterministic, fast). ON DONE: re-run /tmp/r73_analyze_merge_report.sh -> clean scorecard -> finalize -> merge main + push. LESSON: overlay re-bench must match seed COUNT of base data or it Frankensteins. |
| 5 | rebench re-scoped (perf) | 2026-06-15 | First rebench (178500 runs, full 104 graphs, 100 seeds-all) was crawling ~60/min (~44h) -- KILLED. Cause: 100 seeds for DETERMINISTIC mds/pivot (63k wasted) + huge graphs (ba_5000 etc) timing out 300s x3 blocking workers. RE-SCOPED into 2 passes, 89 graphs (<=300 nodes; dropped 16 large floor/insufficient: ba/er/rgg/grids), fast-fail timeouts: PASS A det mds+pivot --seeds 5 (timeout 60); PASS B stochastic umap+fmmm+neato --seeds 100 --seed-refs (timeout 90). Wiped partial dir. Launched pid 2911036 watcher btb1hvf6g -> benchmark_100seed_r73_fixes. ETA ~1-1.5h. NOTE: combos on the 16 dropped large graphs keep r72 verdict (floor/insufficient, fixes dont target them). ON DONE: re-analyze affected-engine combos on 89 graphs (overlay freshest-last) -> scorecard vs 617 -> keep/revert fmmm-M1/neato if net-neg -> summary -> merge develop->main + push. |
| 4 | WAVE 2 done -> Phase B rebench | 2026-06-15 | Wave 2 results HONEST/PARTIAL: fmmm MAARPacking (79a2ac5) source-faithful but M1 mixed/partial (residual=component rotation/trajectory NOT packing); fmmm M5b multi-edge clean win. neato polyomino (786f32b) per-component RMSD~1e-3 but packed spacing differs (residual=edge-spline bboxes), 0 verdict flips. sugiyama coord: 0 VERIFIED FLIPS (needs deep position.c port + mincross rank-seeding; DISCARDED uncommitted, finding documented -- LP-canonical spike = promising future direction). BOTTOM-OF-WELL SIGNAL: wave-2 targets have deep multi-layer root causes, each fix is one layer, verdict only flips when ALL matched. gem: registration kept (already seeded via --seed-refs, harmless, parity-gap deferred). develop r73 commits: 24ae7d3 harness, 7e2f7ae umap, 0c01c00 pivot, 79a2ac5 fmmm, 786f32b neato. PHASE B re-benchmark launched pid 2819559 watcher b62zzmwgc -> benchmark_100seed_r73_fixes (17 reimpl variants umap/mds/pivot/fmmm/neato x 104 graphs x 100 seeds + seeded refs SEEDABLE_BASES). ON DONE: re-analyze (overlay r73_fixes freshest-last on per_combo_r72_final) -> scorecard (vs 617 baseline) -> decide keep/revert fmmm-M1/neato if net-negative -> summary -> MERGE develop->main + PUSH (JMT directive). |
| 3 | WAVE 1 landed + consolidated | 2026-06-15 | Wave 1 committed on develop (0c01c00): umap parallel-edge (7e2f7ae), mds-weight+gem-harness (24ae7d3), pivot scale+unweighted (0c01c00). umap=set->dict accumulator sums parallel-edge mult (verified correct). pivot=skip_normalization + pivot_mds in _UNWEIGHTED_REFERENCE_LAYOUTS. mds=classical_mds in _UNWEIGHTED. **GEM PARITY GUARDRAIL FAILED** (dagua-gem != ogdf-gem at matched seed on benchmark path despite r71's 3.86e-13 claim) -> gem stays divergent, NOT counted as fixed (gem registered stochastic via xfail test; re-bench in Phase B will show its verdict, revert if regresses). INCIDENT: 3 codexes shared one worktree -> git-state collision (pivot left uncommitted, branches tangled); RECOVERED (no data lost), consolidated linear. LESSON: parallel codex needs isolated worktrees OR sequential. classic_fcose test failure = pre-existing drift (not r73). WAVE 2 = SEQUENTIAL: fmmm MAARPacking (pid 2654178 watcher bonnlce8u) -> then neato polyomino -> then sugiyama coord. Specs source-corrected (MAARPacking Best-Fit not TileToRows; polyomino not shelf). ON each DONE: review diff -> next. Then Phase B re-bench (umap/pivot/classical_mds/gem refs + fmmm/neato/sugiyama) -> re-analyze -> scorecard -> summary. |
| 2 | adversarial critique -> de-risked plan; wave 1 dispatched | 2026-06-15 | ADVERSARIAL CODEX (pid 2587782, /tmp/r73_adversarial_findings.md) caught major issues. (1) 3Q-ModeB reclassification = LAUNDERING, REJECTED: proposed rule passes 5/40 chance controls (12.5%>5%); "161" came from inverted convention passing 11/40 (27.5%); mixes loose 5% stress margin w/ strict 2% cross/kNN; no cross-combo FDR (~198 false-equiv under null). PILE 2 (~218) STAYS DIVERGENT. (2) "degenerate=quality-identical" FALSE: ba_500 sugiyama 22344 crossings vs 2805 ref -- genuinely worse. (3) Sonnet packing algos WRONG (read actual source): neato uses POLYOMINO (pack.c polyGraphs, l_node default) NOT shelf; FMMM uses MAARPacking::pack_rectangles_using_Best_Fit_strategy + DecreasingHeight presort + NoGrowingRow NOT TileToRowsCCPacker. (4) sugiyama B+C+D overlap: >=39 B-combos also have A-divergence, max flip ~80. gem needs matched-seed parity test. Source refs at /home/jtaylor/projects/_references/{graphviz,ogdf}. REVISED CEILING: ~100-165 genuine fixes, NO laundering. WAVE 1 dispatched (medium, disjoint files): umap parallel-edge pid 2599479, pivot scale pid 2599681, harness mds-weight+gem-seeded-ref+parity pid 2599889; watcher bl6khmy3x. ON wave1 DONE: adversarial-review diffs -> wave 2 (fmmm MAARPacking, neato polyomino, sugiyama B+C+D) -> Phase B rebench (gem/mds refs need re-run) -> re-analyze -> summary. NOTE: gem+mds change how REFERENCE runs -> require re-benchmark. |
| 1 | R DONE -> P | 2026-06-15 | ALL 6 threads in (findings /tmp/r73_thread{1..6}_findings.md; synthesis /tmp/r73_synthesis.md). PILE 1 FIXES (~171): sugiyama B+C+D ~80 (coord-assign+mincross tie-break), umap parallel-edge 24 (set.add dedups parallel edges vs CSR-sum; HIGH), gem seeded-ref 22 (OGDFGem mis-registered deterministic; mark stochastic+rebench), pivot scale 16 (skip_normalization; RMSD<1e-5), fmmm M1 component-packing 15 (TileToRowsCCPacker not polyomino), neato shelf-packing 3+5, mds-weight 5 (_UNWEIGHTED_REFERENCE_LAYOUTS), fmmm M5b multi-edge 1. PILE 2 QUALITY-IDENTICAL reclass (~218, LAUNDERING RISK): sugiyama-LP-A 135 (degenerate zero-obj LP GLPK vs HiGHS), sfdp 47, mds-degenerate 34, umap 2 -- via 3Q gate change (drop BH, per-combo IUT). PILE 3 FLOOR (~233): sfdp 137, fmmm 72, sugiyama-E 16, drl 5, maxent 3. CROSS-CUTTING: component-packing (neato+fmmm+mds), parallel-edge (umap+fmmm). JMT verified rotation IS handled (Procrustes removes rot/refl/scale/trans; degenerate-MDS = same metric-structure diff arrangement => 3Q not divergent). ADVERSARIAL CODEX critique dispatched pid 2587782 watcher b0uayqu6c (high effort): attack 3Q-laundering (empirically run anti-laundering gate under proposed rule), degenerate=quality-identical claim (does kNN bind?), floor calls, fix-overlap (sugiyama B vs A double-count). ON DONE: de-risked plan -> dispatch fix codexes. |
