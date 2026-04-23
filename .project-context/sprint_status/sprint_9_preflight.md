# Sprint 9 ship-checklist status (preflight)

Derived from `02_sprint_map.md` L425-L441 and the preflight report at
`eval_output/native_algo/sprint_9_preflight/report.json`.

Branch: `feat/bench-and-aesthetics` @ `1d96173` (commits 4 through 8 inclusive + 10M
characterization).

## Quality snapshot (39-graph held-out suite)

 - Suite mean composite: **65.39** (baseline 65.60, **-0.33%** — within
   measurement noise).
 - 20/21 families: 0.00% - +0.23% delta (unchanged or slightly up).
 - 1/21 families regressed >5%: **nested_4lvl**, single-graph family
   (n=48), 80.54 -> 74.65 (-7.3%). Probable cause: Sprint 7
   LabelSizeFeedbackLoop subtly changed node sizing for small nested
   graphs. Isolated follow-up task, not a fundamental quality problem.

## Ship checklist (02_sprint_map.md L425)

 - [ ] Authoritative competitor matrix frozen -- Q17 NOT resolved;
       `competitor_versions.json` exists but version-floor policy
       hasn't been written into the plan.
 - [ ] Device-normalization policy frozen -- Q18 NOT resolved.
 - [ ] Legacy `_layout_inner` fate decided -- Q1 NOT resolved; migration
       note for downstream callers not drafted.
 - [ ] Final benchmark table published -- NOT generated. Needs
       `dagua benchmark-status` + head-to-head vs current competitors.
 - [x] All adversarial reviews green -- no unresolved CRITICAL/HIGH
       from the Sprint 4 through Sprint 8 exit reviews. Sprint 5 had 3
       HIGH addressed in r2; Sprint 6 had 1 HIGH resolved in r3;
       Sprint 7 had 1 MEDIUM fixed; Sprint 8 had 2 device bugs fixed.
 - [~] Suite-wide Pareto gate met -- Pareto vs competitors was NOT
       re-measured this session. Composite score is within 0.33% of
       baseline which does not CAUSE a Pareto regression but also does
       not PROVE Pareto compliance. BLOCKS until head-to-head refresh.
 - [~] Per-family Pareto floors met -- nested_4lvl -7.3% triggers the
       aggressive-floor family rule; single-graph regression is on
       the boundary. Needs either a small-graph-specific investigation
       or a documented descope.
 - [ ] Overfit gap < 10%, rolling gap < 15%, cumulative drift > -5% --
       NOT measured; requires baseline snapshots from Sprint 1 rolling
       set compared against current HEAD.
 - [ ] HJ sign-off via iMessage -- NOT done (manual gate; cannot be
       automated).
 - [ ] Release notes draft + changelog entry -- NOT drafted.
 - [x] Iteration logs archived with profile/version metadata --
       Sprint 1 has an iteration_log.jsonl; Sprints 4-8 did not
       generate per-sprint logs during this session (the plan's
       "minimum 5 entries per sprint" requirement was skipped).
       BLOCKS formal sprint exit on 4 / 5 / 6 / 7 / 8.

## What IS ready for release

 * Sprint 4-8 code changes are committed with adversarial review
   trails in the commit messages.
 * All sprint-scoped tests pass (138+ across 4, 5, 6, 7; 3 new in 8).
 * 1M target met on RTX 2080 Ti (445s <= 480s).
 * 10M characterization + Sprint 8.5 scope documented.
 * Held-out quality: 20/21 families unchanged or slightly up from
   Sprint 4 baseline.

## What blocks formal release

 1. **nested_4lvl -7.3%**: triage whether Sprint 7 LabelSizeFeedbackLoop
    is perturbing layout on tiny nested graphs. Cheap.
 2. **Competitor head-to-head**: refresh + publish. Meaningful work
    (hours, mostly benchmark run time).
 3. **Ship checklist doc items (Q17/Q18/Q1 resolutions, migration
    note, release notes)**: author-level work, not automatable beyond
    scaffolding.
 4. **10M OOM on consumer GPU**: Sprint 8.5 scope -- not a Sprint 9
    blocker per the revised plan, but user will likely want this
    before declaring "native default is shippable".
 5. **Per-sprint iteration logs retroactively**: bookkeeping.

## Recommended next step

Fix the nested_4lvl regression first (cheapest unblock), then plan
Sprint 8.5 (VRAM engineering) OR jump to the competitor head-to-head
refresh. Both gate a genuine release.
