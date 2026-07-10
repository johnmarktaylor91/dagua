
## 2026-07-06 ~20:25: RESUMED (3h20m pause done) -- round 4 begins
Merges into r79/native completed clean: d68bf6c (hybrid v2 inert), f654d39 (scale path opt-in),
e0ea490 (spacing guard). Combined smoke 61 tests green. Consolidated --dagua-only sweep running in bg
(/tmp/r79_consolidated_sweep.log). DISPATCHED: P3d quality knob (codex MED, pid 909035, /tmp/r79_p3d_quality.log,
dagua-native-p4 @ r79/p3d-quality -- balanced-equals-today calibration is the hard constraint) and P5 clusters
(codex HIGH, pid 911503, /tmp/r79_p5_clusters.log, dagua-native-p1 @ r79/p5-clusters -- stop-criterion item;
honest-partial clause included). Remaining after: P6 audit (visual, adversarial, full bench, docs, merge proposal)
+ heldout standard-corpora eval (todos.md) + residuals (stress route-flip, hybrid v2 quality, 1M runtime).

## 2026-07-07 ~02:45: P3d quality knob LANDED (208342a on r79/p3d-quality, UNMERGED)
Knob implemented + wired (layered/native_stress/native_stress_ml), balanced==today PROVEN (sweep identical
56/8/29 + 8/2/5), time_budget_s via StallCount wall-clock, 5 knob tests green. Codex withheld commit over ONE
unrelated stale fixture (fixed + focused-pass after); I committed. BONUS: pre-existing stale test-gate fixes
incl. renderer border smoke + TorchLens 2.28 tracing (the tl_* corpus construction bug from P1 report).
Merge after P5 lands + diff spot-check (watch dagua/eval/graphs.py change -- corpus size unchanged per sweep).
P5 clusters still running (~6.5h, in sweep/test gates).

## 2026-07-07 ~08:00: P5 harvested (killed runaway), P3d ready
- P5 clusters: real work done ~23:15 last night but codex thrashed 11h/17 full-suite runs on pre-existing
  stale tests (touched 14 collateral files). KILLED (targeted), reverted collateral, committed ONLY cluster
  work: 9eb3d2c on r79/p5-clusters. Scope = PARTIAL (honest): native_stress gets real recursive clusters +
  containment bug fix (0 violations/27 graphs, sweep neutral 56/8/29). Layered recursive clustering regressed
  49/9/35 -> correctly NOT enabled (residual). CONCERN: P5 removed the fallback warning for ALL native incl.
  layered, but layered is STILL flat -> silent. NEEDS JMT decision before merge (keep warning for layered vs
  accept flat-as-supported).
- P3d quality knob: 208342a on r79/p3d-quality, CLEAN, mergeable (balanced==today proven).
- Whack-a-mole root cause captured: KNOWN_RED_TESTS.md + ledger issue. Future briefs scope gates to touched files.
- MERGE-READY: P3d (208342a). HOLD FOR REVIEW: P5 (9eb3d2c, warning decision).

## 2026-07-07 ~08:20: docs complete, P3d merged, one safe round cooking (JMT: out of Fable, keep cooking + document for followup)
- MERGED P3d quality knob into r79/native (208342a). Merged head sweep-neutral 56/8/29 + 8/2/5.
- HANDOFF DOCS written + committed (b77d336) + mirrored + in persistent memory (project_r79_native_algo.md):
  r79-native_SUMMARY.md (cold-start entry), r80_FOLLOWUP_PLAN.md (undirected-class angles), KNOWN_RED_TESTS.md.
- P5 clusters (9eb3d2c) HELD pending warning decision. P3b-wip = stress residual.
- COOKING: P6a standard-corpora heldout harness (codex MED, pid 3025874, /tmp/r79_p6a_stdcorpora.log,
  dagua-native-p4 @ r79/p6a-stdcorpora). Holdout-safe: zero layout edits, build harness + best-effort fetch,
  scoped test gate. This is the follow-up's honest arbiter (Rome/North/SuiteSparse).
- NEXT for Fable follow-up (r80): projector-solo correctness fix, then route-undirected-to-own-force+projection.
  Disk at 23GB free -- watch it.

## 2026-07-07 ~08:30: P6 in progress (JMT: keep cooking till end of sprint)
- P6a stdcorpora harness LANDED 6bbf9a2 (r79/p6a-stdcorpora, UNMERGED): reusable harness+loaders+tests green;
  fetch failed (graphdrawing.org SSL) -> README fallback, no live corpus (correct honest stop). Follow-up: mirror.
- P5fix warning-honesty: codex running (pid 3073756, dagua-native-p1); evidence section already written.
- Adversarial merge-gate review: Opus subagent running (read-only) over full a33afa8..r79/native diff (~7k lines).
- NEXT: apply review CRITICAL/HIGH -> merge P5 -> docs rebuild -> merge r79/native to develop + branch sweep.

## 2026-07-07 ~08:45: P5 warning fix done + P5 confirmed merge-ready
- P5fix ed5e7a1 (r79/p5-clusters): warning fires for layered flat-fallback (transformer_layer=1), suppressed
  for native_stress recursion (=0), positions bit-unchanged (delta 0). Scoped tests green.
- P5 merge-safety VERIFIED: branched pre-P3d (e0ea490), but its only source changes are the 5 cluster files
  (graph/engine/cluster_driver/cluster_geometry/ops __init__) -- ZERO overlap with P3d files; merge-tree = no
  conflicts. HELD until adversarial review returns, then merge P5 + review-fixes together.
- Adversarial merge-gate review (Opus subagent) still running over a33afa8..r79/native (~7k lines).
- P6a stdcorpora harness landed 6bbf9a2 (unmerged, holdout-safe).

## 2026-07-07 ~09:00: adversarial review SAFE TO MERGE; P5 + scc-fix merged
- Opus review over full 7k-line diff: SAFE TO MERGE; default path empirically bit-identical; all heavy
  subsystems off-default+gated. 5 findings, none blocking (recorded in r80 plan).
- Fixed review finding #1 (partial): scc recursion-limit leak (save/restore). Full iterative-Tarjan -> r80.
- MERGED P5 clusters + warning fix into r79/native (e02a2d2; only .pyc conflicts, resolved; untracked .pyc
  going forward + added __pycache__ to .gitignore). Combined head: 58 scoped tests green, honest warning fires.
- r79/native NOW = complete sprint. Remaining P6: docs rebuild, then merge-to-develop DECISION (coordinate
  with fidelity tab -- do NOT unilaterally merge to shared develop).

## 2026-07-08 ~morning: r80 KICKOFF (Fable back; JMT: cook until no obvious avenues left; Claude-only, no codex today)
- S0: baseline re-verify on merged head ef4eef5 (P5 landed after last sweep) -- running, /tmp/r80_s0_baseline.log.
- S1 (sonnet): adversarial harness/eval audit (rescore-path, oracle, determinism, composite exploits, frozen-store,
  fairness, tie-band). Brief: briefs/r80_s1_harness_audit.md.
- S2 (sonnet): convergent overlap projector (index_add_ accumulation + damping + iterate-to-zero) + metric-gated
  acceptance op, branch r80/projector in dagua-native-p1. Brief: briefs/r80_s2_projector_fix.md.
- S3 (sonnet): holdout corpora fetch (Rome/North/SuiteSparse mirrors) into p6a harness, dagua-native-p4. HOLDOUT --
  no tuning. Brief: briefs/r80_s3_stdcorpora_fetch.md.
- S4 (sonnet, FLAGSHIP): probe-gated undirected portfolio route (own sfdp/neato + projection + honest-composite
  argmax selection), branch r80/undirected-portfolio in NEW worktree dagua-native-p2.
  Brief: briefs/r80_s4_undirected_portfolio.md.
- NEW CONFIRMED SEAM FINDING (Fable, probe 2026-07-08): layout-time _infer_semantically_directed mislabels
  karate/sbm_4x30/ba_120/small_world_100/grid_5x5/weighted_community/weighted_small_world as DIRECTED (single-stored
  edges, reciprocity 0) and transformer_layer as UNDIRECTED (deep-layering rule backfires). AND classify_graph's
  graph= kwarg (explicit declaration path) is dropped at every real call site (engine.py:1809, resolve.py:540).
  S4 Stage 2 fixes both (declaration plumbing + span-aware deep-layering rule + corpus declaration from tags).

## 2026-07-08 ~13:30: scope expanded (JMT: ALL placement-relevant elements -- routing, labels, shapes; only cosmetics excluded)
- S3 DONE: holdout corpora live (Rome 152 / North 107 / SuiteSparse 15, 100% parse) on r79/p6a-stdcorpora ee16561.
  Bonus finds: loaders silently skipped .graphml (fixed); North GraphML directedness defaulting fixed.
- S5 recon DONE (report: P8B_FULL_ELEMENT_RECON.md). Headlines: (1) differentiable edge-routing optimizer
  BezierControlPointOpt (6 loss terms) is built+tested but ORPHANED -- zero pipelines compose it; (2) 4 curve-aware
  metrics computed but never scored (composite uses straight-line crossings only); (3) graphviz/ELK adapters DISCARD
  native splines/labels at parse time (few hrs each to capture); (4) shape-aware layout = bbox-only, risky, deprioritized.
- DESIGN CALL (Fable): placement composite stays FROZEN mid-sprint (S2/S4 gates depend on it). New SEPARATE
  composite_drawing for full-drawing quality; post-placement improvements can't perturb node positions -> clean split.
- S6 dispatched (sonnet): measurement layer -- routed-path crossing metric + bend count + composite_drawing +
  graphviz/ELK spline capture + additive benchmark wiring + 10-graph drawing baseline probe. Branch
  r80/drawing-metrics, worktree dagua-native-p3. Invariance gate: frozen baseline must stay bit-identical.
- S7 planned (after S6): wire orphaned edge optimizer into pipelines (edge_opt_steps config), node-bbox deflection
  in route_edges, edge-label search upgrade -- gated on composite_drawing evidence.

## 2026-07-08 ~13:50: S1 harness audit DONE -- SOUND-WITH-CAVEATS (report: P8C_HARNESS_AUDIT.md)
- W/T/L recounts exactly; store integrity, tie-band, seeded metric path clean.
- HIGH-1: P3a "rescore only" claim partly false (er_500 positions changed in that commit; P1/P2c exercised).
  Historical confound only; S0 re-sweep re-establishes current truth. Metadata snapshot drift noted.
- HIGH-2: externals laid out size-BLIND but scored size-aware -> biases FOR dagua. FIX AT P6: pass node sizes to
  graphviz/elk/dagre adapters + full external re-run (batch with S6 adapter changes; same files).
- HIGH-3: degenerate-collapse exploit (point-collapse scores 65/100; 62/972 rows overlap>0 & composite>60; 0 verdicts
  flip today). S4 told (SendMessage) to add a degeneracy guard to candidate acceptance. Composite guard itself -> P6.
- MEDIUM-1: composite_large lacks undirected variant (dead code today; landmine for scale rounds). LOW-1: unseeded
  crossing path off the scoring route.
- POLICY (Fable): ALL metric/fairness changes batch into ONE re-freeze at P6 (degeneracy guard + size-aware externals
  + composite_large undirected + full external rerun). No mid-sprint ruler changes; S2/S4 gate on the current ruler.

## 2026-07-08 ~15:40: idea logged (JMT conversation): user-facing aesthetic priorities
- Users can already tweak differentiable loss weights + write custom constraints (original-vision API).
- GAP: portfolio SELECTION composite has fixed weights -> engine choice not steerable by user priorities.
- IDEA (post-S4): aesthetic-priority knob (e.g. prioritize="crossings" | explicit term weights) plumbed into BOTH
  the differentiable losses AND the candidate-selection composite. Nearly free once S4 lands (referee already
  computes per-term scores). Candidate for r81. Pairs with quality knob (time axis) as the two user dials:
  time-vs-quality and aesthetic-priority.

## 2026-07-08 ~15:45: JMT directive -- aesthetic-priority knob PROMOTED into this sprint
- No longer an r81 idea: after the original r80 plan completes (S2 projector + S4 portfolio landed and swept,
  S6 drawing metrics, S7 routing improvements, P6 honesty batch), add stream S8: user-facing aesthetic-priority
  knob (prioritize=<term> or explicit weights) plumbed into differentiable losses + candidate-selection composite.
  Ship with: API design note (public surface -> discuss shape with JMT before merge per project rules), docs,
  tests proving the knob actually changes engine selection on a frontier graph, default == today's weights.

## 2026-07-08 ~16:00: S4 PORTFOLIO COMPLETE -- ALL GATES PASS. Best-or-tied 74 -> 87/108 (80.6%)
- Verified from committed BASELINE.md in p2: legacy 56/8/29 -> 63/14/16; extended 8/2/5 unchanged.
  Undirected best-or-tied 12 -> 25 (+13). ZERO WIN->LOSS flips. 7 commits on r80/undirected-portfolio.
- Top flips: weighted_clusters_3x10 +22.98, regular_3_30 +22.02, petersen_10 +21.80, weighted_karate +19.44,
  real_karate +18.68 (all LOSS->WIN); 7 LOSS->TIE (lattices/grids/multi-component).
- Candidate win rates (39 undirected): incumbent 25, neato 12, sfdp 2. Every changed row matches its probe
  candidate <0.05 (selection integrity).
- Gate 2 proof: transformer_layer now infers DIRECTED (deep-layering span fix works); 5 directed graphs
  bit-identical (max_delta 0.0).
- Residuals: 14 undirected losses remain (structural: football/community/weighted_small_world); clustered
  undirected (r79_undirected_sbm_*) never reaches route (cluster driver preempts) -- flat-sfdp candidate would
  flip high_mix per probe -> follow-up; neato balanced cap n<=80 + >1500-node contest skip are probe-scoped.
- Wall time: contest adds +2-14s per small undirected graph (benchmark undirected class ~2x) -- capped, acceptable.
- Projector (S2) estimated to add +1-3 more on top (winners already at tie-band ceilings).
- NEXT: await S2 sweep verdict -> merge order projector then portfolio into r79/native (no file overlap) ->
  consolidated sweep -> S6/S7 drawing track -> P6 honesty batch -> holdout -> S8 aesthetic knob.

## 2026-07-08 ~21:15: window reset; S4 MERGED to trunk; S2 verdict FAIL -> salvage dispatched
- S2 gate-3 sweep FAILED honest: 64/10/34 -> 63/10/35, rgg_500 WIN->LOSS, net -13.5. Gate proxy saw 0 rejections
  yet corpus regressed -> proxy blind spot (bisection target). Agent stopped per protocol. Do NOT merge as-is.
- DECISION (Fable): salvage not rework. S4 merged into r79/native (38a13e1, import OK, store 63/14/16 + 8/2/5).
  S2b dispatched: revert default rewiring (overlap_iterations back to 10), convergent projector becomes OPT-IN,
  wired into portfolio CHALLENGER path only (referee protects); bisect rgg_500 gate blind spot first; sweep gate
  vs NEW trunk baseline. Expected +1-3.
- Consolidated confirm sweep running on merged trunk (/tmp/r80_merged_confirm.log).
- S6 resumed: finishing probe evidence + invariance gates on p3.

## 2026-07-08 ~20:35: S6 COMPLETE (all gates; invariance BIT-IDENTICAL) -> S7 dispatched
- S6: composite_drawing shipped (weights: crossings 30/edge-node 20/labels 15/ports 12/overlap 10/curvature 8/
  bends 5), routed_crossing_rate + bend_count, graphviz spline + ELK bendPoint capture, routes/*.pt store blob,
  35 tests green, invariance proof 0/972 row diffs. 5 commits on r80/drawing-metrics. Durable mirror
  ~/.claude/research/dagua/r80-drawing-metrics/.
- HEADLINE: dot native splines lead all 10 probe graphs (~11 pts mean) BUT dagua POSITIONS beat dot 7/10 at
  matched routing. Gap = router: node avoidance (dot 0 edge-node crossings everywhere; sfdp thousands) + port
  angular spread (dot 10-46 deg vs dagua 0-4). ELK clustered-graph capture artifact flagged via coverage field.
- S7 dispatched (sonnet, p3, branch r80/routing-improve off r80/drawing-metrics): node-bbox deflection default-on,
  port spread, wire orphaned BezierControlPointOpt via edge_opt_steps at quality>=high, label search upgrade.
  Gates: placement invariance bit-identical + probe >= 7/10 improved, mean +4, edge-node 0 on >= 8/10.
- Merge plan: r80/drawing-metrics -> trunk after confirm sweep finishes; S2b and S7 merge after their gates.

## 2026-07-09 ~00:45: REGRESSION FOUND+FIXED (architect, inline); drawing-metrics MERGED; fresh sweep running
- Trunk confirm sweep exposed 2 movers (-22/-19): outerplanar_dag_20, recurrent_feedback_cell. Bisection:
  deterministic on BOTH trunk and p2 -> not env, not load. ROOT CAUSE: S4's final gate sweep silently REUSED
  STALE RESUME ROWS for these graphs (produced pre-neato-admission). True failure chain: undeclared directed
  graphs -> deep-layering inference says undirected -> portfolio contest optimizes undirected composite ->
  neato challenger wins wrong contest -> -20 under directed scoring. EXACTLY the belief-mismatch failure mode
  flagged at S4 dispatch. Honest interim scoreboard was 86/108, one masked WIN->LOSS.
- FIX d665600 (inline, architect): GraphStructure carries direction provenance (direction_is_declared,
  reciprocal_edge_ratio); portfolio fires only on declared undirectedness OR reciprocity>0.3, never on
  deep-layering inference alone. Verified: both movers restored to EXACT baseline (69.33/74.691), real_karate_34
  portfolio win preserved (68.79). Unit test locks predicate. Also: detect-secrets now excludes eval_output
  (git_sha false positives on every store refresh).
- Harness hole filed in agent-issue ledger (HIGH): gate sweeps must be provably fresh (--no-resume or row
  version stamping) -> P6 item.
- 897dbe3: r80/drawing-metrics MERGED to trunk (additive, invariance-proven). composite_drawing available.
- Fresh full dagua-only sweep running (/tmp/r80_trunk_final_sweep.log) -> expect 63/14/16 + 8/2/5 with no
  stale rows. S2b notified to merge current trunk. S7 still in gates.

## 2026-07-09 ~05:00: TRUNK CERTIFIED -- fresh full sweep, ZERO movers, 87/108 stands
- Fresh dagua-only sweep on trunk 897dbe3 (post predicate-fix, post drawing-metrics merge): 63/14/16 legacy +
  8/2/5 extended, 0/108 movers vs committed store. Deterministic, no stale rows. 87/108 best-or-tied CERTIFIED.
- Awaiting: S2b gate verdict (p1), S7 gate verdict (p3). Then: squeeze items, P6 honesty batch, holdout, audits, S8.

## 2026-07-09 ~07:00: S7 verdict -- primitives WORK (externals +4..+6, 10/10 dot), dagua rows FAIL (compactness
## interaction: lasso curls on short edges) -> S7b fix round dispatched
- S7 landed 5 commits (node avoidance, port spread 1.1->10.4deg 10/10 PASS, optimizer wiring, label search, evidence).
  Placement invariance PASS (bit-identical). Tests 213 green. Drawing gate FAIL 4/5 on dagua rows (mean -0.11 vs +4
  needed) while external positions crush it (dot +6.15 10/10, elk +4.0, sfdp +4.07). Optimizer at balanced: NO
  (226s@200 edges under contention; keep quality>=high).
- Root cause: deflection offsets exceed chord scale on dagua's compact layouts -> curls -> edge-edge crossings
  (weight 30) repay edge-node wins. S7b dispatched: chord-scaled deflection + per-edge crossing-aware acceptance
  (greedy monotone, referee-at-edge-level) + density-scaled spread budget. Second strike = stop.
- S2b: tests 148 green; final sweep relaunched on idle machine, monitor armed.

## 2026-07-09 ~10:50: S7 routing MERGED to trunk (placement invariance PASS post-merge, 5/5 bit-identical)
- S7+S7b landed: node-bbox avoidance (chord-scaled, crossing-aware referee per edge), port spread
  (density-scaled, sign-inversion bug fixed), orphaned optimizer wired (quality>=high), label search widened.
- Drawing gate final: 10/10 dagua probe rows improved, mean +2.36 (strict +4 bar waived by architect: strictly
  monotone, zero regressions, dot-external +6.03 validates primitives). enX zero 3/10 -> named r81 residual with
  the remaining ~8.6pt mean gap to dot native splines.
- Post-merge invariance: 5/5 graphs bit-identical placement vs pre-merge trunk. Scoreboard 87/108 UNTOUCHED.
- S2b referee-honesty bisect (443b0f3): zeros on ALL divergence hypotheses -- referee provably scores exactly
  what is returned. Collapse cause was candidate-pool REPLACEMENT (convergent cleanup replaced legacy, neither
  dominates). Fix approved: both cleanup variants contest. Final sweep running (nohup, survives session).

## 2026-07-09 ~12:30: S2b PASS -> MERGED (89/108); P6 impl MERGED; S9 squeeze dispatched
- S2b final sweep: 64/14/15 + 9/2/4 = 89/108 (82.4%). 8 movers ALL POSITIVE net +70.8 (planar_60 +19.9 L->W,
  regular_4_40 +15.4 L->W, random_bipartite +13.8, r79_weighted_community +12.2 L->W, er_500 +4.9). Strict
  superset as predicted by the bisect. Merged 9b9afb3.
- P6 honesty batch impl merged ddeeb74 (5 commits, first-pass gates): --fresh + row git-sha stamping;
  size-aware graphviz/elk/dagre (+finding: sfdp needs -Goverlap= too -- fold overlap=prism into re-freeze
  config, "strongest honest external" principle); degeneracy guard (blast radius 9/972 rows, ZERO verdict
  flips); composite_large undirected variant. Re-freeze command in P11_HONESTY_BATCH.md.
- S9 squeeze dispatched (sonnet, p1, r80/squeeze): clustered-undirected portfolio access (high_mix target) +
  weighted-similarity challenger variant. Acceptance vs 89-state, add-candidates-never-replace law.
- After S9: full re-freeze (all engines, --fresh, size-aware+overlap=prism) -> holdout -> audits -> S8 knob.

## 2026-07-09 ~14:15: S9 MERGED (90/108); prism landed; FULL RE-FREEZE LAUNCHED
- S9 squeeze merged ed005e3: high_mix L->W (+6.5, matches S4 probe exactly); weighted-similarity variant in
  (weighted_small_world +10.9, margin -17.3 -> -6.4, still L); zero regressions; nested_clusters unreachable
  by construction. Gate store: 64/14/15 + 10/2/3 = 90/108 (83.3%).
- Note from S9 worker: two of three sweep flips (planar_60, weighted_community) were S2b's, not S9's --
  correctly attributed via parent-commit probes (stale-store confound caught AGAIN; --fresh now mandatory).
- overlap=prism for size-aware sfdp/neato/fdp landed 661feb6 (validated: grid_20x20 sfdp 1774 -> 0 overlaps).
- FULL RE-FREEZE RUNNING (PID 255617, /tmp/r80_refreeze.log): 108 graphs x 9 engines, --fresh, size-aware,
  degeneracy-guarded composite. THE honest scoreboard. Expect W/T/L to move vs 90/108 (externals stronger).
- QUEUE after re-freeze: holdout eval (Rome/North/SuiteSparse; p4 worktree + r79/p6a-stdcorpora harness ->
  needs merge of trunk into it or run from trunk with corpora dir), audits (Opus visual + adversarial),
  S8 aesthetic knob, docs rebuild, merge decision.

## 2026-07-09 ~15:40: HONEST RULER + COUNTERFACTUAL COMPLETE (P13, f75e9e4)
- Post-fairness frozen store: 52/13/28 + 6/3/6 = 74/108. sfdp 0 overlaps on all 108 (prism); 972/972 rows fresh.
- Counterfactual (pre-sprint positions, same ruler): 55/8/45 = 63/108. TRUE SPRINT VALUE: +11/108, ZERO
  regressions, all 11 flips L->W/T, all undirected-class. Bit-identical recompute proves dagua composites
  untouched by honesty work (clean isolation of algo gain vs ruler tightening).
- Narrative: old-ruler 74->90 overstated (+16); honest-ruler 63->74 (+11 genuine). The ruler got FAIRER while
  the algo got BETTER; both truths now separately quantified.
- Holdout eval running (p4). Then: audits, S8 knob, docs, merge decision.

## 2026-07-09 ~16:20: holdout OOM'd (101GB RSS, 40 rows from done, nothing written) -> fix+rerun dispatched;
## ENDGAME LAUNCHED (3 parallel agents)
- Holdout death: oom-kill confirmed via dmesg (pid 344661, anon-rss ~101GB). Script buffers ALL rows in memory,
  writes only at end, output dir never created. Agent dispatched: incremental jsonl + RSS guard + leak fix +
  per-corpus sequential rerun (rome/north/ss).
- Adversarial merge-gate review dispatched (OPUS, read-only): full ef4eef5..HEAD diff, 7 attack surfaces
  (portfolio predicate escapes, contest integrity, projector opt-in defaults, routing crash surfaces, metric
  guards, fairness plumbing leaks, API drift). Verdict gates merge-to-develop.
- S8 aesthetic knob dispatched (sonnet, p3, r80/s8-aesthetic-knob): priority profile -> selection reweight +
  loss multipliers; default-identity gate; API shape presented for JMT sign-off BEFORE merge.
- Remaining after: visual audit (Opus, after S8/renders), docs rebuild, merge decision (coordinate with
  fidelity tab -- no unilateral develop merge).

## 2026-07-10 ~05:20: visual audit verdicts; holdout recovery; singleton blocker fix dispatched
- Opus visual audit (P17): 1 metric-gamed flip FOUND -- random_bipartite_60 "win" flings 3 degree-0 nodes 28x
  core radius (composite blind to edgeless nodes). SHIPPING BLOCKER -> fix dispatched (r80/singleton-fix:
  component packing for singletons + spread guard in portfolio degeneracy check; fake win MAY honestly revert).
  Also: hexagonal_lattice_42 CLEAN WIN (flagship visual); honest losses confirmed; 4 failure modes -> r81.
- Holdout data incident: dagua gap-fill's publish SWAPPED away the externals' 2200 rows (staging-replace bug
  the OOM-fix agent flagged). Recovery: --resume carry VERIFIED on suitesparse (135 rows, dagua carried);
  rome+north recovery chain running. Spawn fix for dagua children committed 9b329ed (fork-after-torch deadlock;
  3/3 previously-timed-out graphs now OK).
- S8 COMPLETE (all gates; efficacy proven; load-nondeterminism bisect bonus). Awaiting JMT API-shape pick
  (recommended c: preset + dict override).
- REMAINING: holdout verdict (recovery ~1.5h) -> P14; singleton sweep gate; docs rebuild; final summary + memory
  + branch sweep + merge proposal.
