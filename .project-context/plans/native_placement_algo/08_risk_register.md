# Risk Register

Risks the plan must survive. Each has a mitigation, a detection signal, and
an abort criterion. "Abort" means roll back the current sprint and escalate
to the user.

## R1 -- Frankenstein risk (integrating hybrid classical + differentiable)

Description: Adding warm-starts, layer-sweeps, Brandes-Kopf, and similar
classical steps to a differentiable core risks creating an unreadable,
non-composable engine with hidden state coupling and "magic" ordering.

Mitigation:
- Every classical step is a registered op. No inline helpers in the pipeline.
- SolveState keeps its 9 typed fields. Algorithm-specific state goes in
  `extras` with the `algo_field` convention.
- Sprint 3 is the Frankenstein-prone one; it has a mandatory mid-sprint
  adversarial review with Frankenstein focus (see 06 prompt template).
- Naming discipline: each classical step op carries the algorithm family in
  its class name (e.g., `SugiyamaLayerSweep`, `BarycenterRefine`).

Detection:
- `dagua/layout/ops/pipelines/dagua_native.py` grows past ~400 LOC.
- SolveState gets new typed fields.
- Adversarial review flags inline helpers.

Abort:
- Sprint 3 adversarial review BLOCK with Frankenstein finding. Rollback
  Sprint 3, replan with smaller scope.

## R2 -- Overfit to iteration suite

Description: Iterating against the fixed iteration suite until scores look
great on it, then discovering held-out or real-world graphs regress. This is
the default failure mode of metric-driven tuning.

Mitigation:
- Held-out suite never touched mid-sprint.
- Rolling random-generator set per sprint exit.
- Anti-overfit gap check at Sprint 9 (iteration - holdout < 10%).
- No coefficient tuning until Sprint 9; sprints 1-8 only add structure.

Detection:
- Iteration composite improves but held-out composite does not (>3% gap
  sprint-over-sprint).
- Rolling-set scores lag held-out consistently.

Abort:
- Two consecutive sprints with growing overfit gap -> roll back to the last
  known-good sprint and take a replan with smaller iteration suite overlap.

## R3 -- Scaling cliff at 1M+ nodes

Description: Layouts look great at 10K. At 100K, memory blows up (autograd 3-4x
multiplier). At 1M, nothing fits. At 10M, we cannot even coarsen in-memory.

Mitigation (revised per 2026-04-22 adversarial review):
- Scaling principles doc is authoritative -- see
  `.project-context/knowledge/scaling_principles.md`.
- **Sprint 1 MUST port per-loss backward + checkpointing + hybrid device ops**
  from legacy `_layout_inner` to the ops pipeline. The draft assumed these
  would be free-standing; they are not. Sprint 1 exit gate grew a
  memory-parity criterion accordingly.
- Sprint 2 fixes the ops hierarchy builder's no-offload + copy-every-tensor
  behavior (see coarsen.py:643-664, 974-988). No-copy hierarchy transfer and
  ops-native disk offload are Sprint 2 exit criteria.
- Pre-dispatch pre-flight checks: RAM/VRAM/disk numbers are reported, never
  guessed. See CLAUDE.md OOM rules.
- Coarsening offload path (disk for 1B-node footprint) is partially plumbed.
  Specifically: `edge_index` and `node_sizes` offload, but
  `fine_to_coarse`, `fine_layer_assignments`, `coarse_layer_assignments` stay
  resident. See R11 for the follow-up.
- Until the Sprint 1 port ships, Mega and Ultra runtime budgets in 03 are
  STRETCH targets, not exit criteria. Sprints 2-7 must not claim success
  based on Mega/Ultra runtime until Sprint 8 verifies it.

Detection:
- OOM at any tier that should fit.
- Peak RSS >1.5x the stated budget.
- Runtime >2x the stated budget.

Abort:
- OOM at the Large (10K) tier -> abort the sprint. Scaling should be free at
  10K. This indicates a deeper regression.
- 3 consecutive tiers over budget -> abort, retrospective.

## R4 -- Benchmark harness drift

Description: The iteration harness or held-out suite is "updated" during a
sprint to accommodate a feature change, unintentionally invalidating all
prior baselines.

Mitigation:
- Held-out is immutable. Iteration suite additions must bump
  `iteration_suite_version`.
- Rolling-seed strategy is deterministic per sprint tag; the code that
  consumes it is tested.
- Baseline JSON files are committed; git catches accidental mutations.

Detection:
- Baseline JSON content changes between sprints without explicit version bump.
- `iteration_suite_version` in `dagua/graphs/iteration/VERSION` moves without
  an exit note reference.

Abort:
- Held-out mutation -> immediate abort of the sprint, git revert, retro.

## R5 -- Codex spec rot

Description: Codex specs reference files that have moved or been deleted
between planning and execution. Codex writes to wrong locations or fails
silently.

Mitigation:
- Every Codex spec runs through a pre-flight grep to confirm target files
  exist at stated paths.
- Specs include a sprint-file link; Codex is asked to cite back the sprint
  file in its output.
- Codex `completeness_contract` block mandates listing all edited files.

Detection:
- Codex completes "successfully" but the claimed edit is not in git diff.
- Test suite passes but the feature under test is not present.

Abort:
- Two consecutive Codex runs exhibiting spec rot -> pause, audit spec
  templates, fix pattern before next dispatch.

## R6 -- Hidden state coupling in ops

Description: A registered op silently reads or writes module-level state
(globals, class variables, singletons) outside SolveState. Composability
lie -- pipelines look composed but actually depend on order-of-imports or
monkey-patched state.

Mitigation:
- Sprint 0 adversarial review with a focus specifically on hidden coupling.
- All new ops have a determinism test: construct two SolveStates, call the
  op, compare output.
- `dagua/layout/ops/__init__.py` reviewed for side-effects.

Detection:
- Op output varies with the order other ops were imported.
- Two consecutive runs of the same pipeline produce different results without
  an RNG change.

Abort:
- Finding of hidden coupling in Sprint 0 -> abort Sprint 0 until fixed. Not
  optional.

## R7 -- Aesthetic-metric divergence

Description: Quantitative metrics improve but the layouts look worse to a
human. Purchase (1997) prioritizes the right criteria but we might encode
them wrongly.

Mitigation:
- HJ ping at every sprint exit (one 3x3 grid). Rate-limited but not skipped.
- Adversarial agent pair reviews images at sprint exit (Codex + Claude
  subagent); disagreement >1 point triggers HJ.
- Flagship-graph set per sprint file tracks the hardest-to-visualize cases.

Detection:
- HJ rating <=3/5 when composite score >= previous sprint's.
- Adversarial agents disagree with composite.

Abort:
- Two consecutive sprints with HJ <=3/5 -> pause, reweight composite per
  rubric change protocol (Sprint 9 activity moved early).

## R8 -- Legacy engine removal breakage

Description: Deleting or archiving `_layout_inner` breaks downstream TorchLens
or notebooks that imported it directly.

Mitigation:
- Sprint 0 Task 0.3 waits on user answer to 09 Q1.
- If the user says "archive," keep an import shim at
  `dagua.layout.engine` that re-exports from `_archive/`.
- CI includes a TorchLens smoke test before and after.

Detection:
- TorchLens import fails post-Sprint-0.
- `git grep _layout_inner` returns usages outside `_archive/`.

Abort:
- TorchLens breakage -> revert Sprint 0 Task 0.3. Keep the shim.

## R9 -- Dispatch reliability (Codex background silently no-op)

Description: From CLAUDE.md: `Skill(codex:rescue)` silently no-ops in some
session-resume conditions. We lose a sprint-hour to waiting on a run that
never started.

Mitigation:
- Prefer direct CLI via Bash with `run_in_background=true`, per CLAUDE.md.
- Verify within 8s with `pgrep -fl "codex exec"`.
- Every dispatch turn's FIRST action next turn is a verify, not a poll.

Detection:
- No `codex exec` process in `ps` after 8s.
- Expected output file missing after expected time.

Abort:
- Redispatch. If two successive dispatches fail, escalate to user.

## R10 -- Plan scope creep

Description: Once drafting is done and sprints begin, scope "grows" --
adding Sprint 10 "generative AI layout"; adding a new theme engine; adding
ISV-specific features.

Mitigation:
- Non-Goals list in 00_overview.md. Any exception must be documented in a
  new Sprint "10+" plan with user sign-off.
- Sprint exit notes include a "scope discipline" line: any work that went
  beyond the sprint's declared goal.

Detection:
- Sprint exit note mentions "also did X" where X was not declared.

Abort:
- Two sprints in a row with scope leaks -> pause, retro, tighten sprint
  file language.

## R11 -- Hierarchy metadata resident during offload (NEW)

Description: Legacy multilevel offload saves `edge_index` and `node_sizes`
but leaves `fine_to_coarse`, `fine_layer_assignments`, and
`coarse_layer_assignments` tensors in memory across all levels. Documented
in gotchas. At 1B-node scale, this alone is ~22 GB of residency.

Mitigation:
- Sprint 8 exit requires ops-native offload of all three hierarchy metadata
  tensors via the save/load manifest mechanism.
- Test: post-offload RSS drops as expected; post-reload RSS returns to
  pre-offload within 5%.

Detection:
- Offload byte-count claim does not match measured RSS drop.
- Reload path needs tensors that were never saved (legacy bug from gotchas).

Abort:
- Sprint 8 exit gate; metadata offload failure blocks exit.

## R12 -- Cumulative regression drift (NEW)

Description: A per-sprint -3% composite tolerance compounds to -24% to -27%
across nine sprints. Each exit passes but the final default is materially
worse than Sprint 0 baseline.

Mitigation (from adversarial finding in 2026-04-22 review):
- Cumulative bar added in 03: composite cannot drop more than -5% absolute
  vs Sprint 0 baseline without explicit user waiver.
- Per-family veto bars added in 04: near_clique/disconnected/pathological
  have their own floors that cannot be breached regardless of aggregate score.

Detection:
- `scripts/quality_runtime_analysis.py --sprint-exit` computes cumulative
  drift vs Sprint 0; fails exit if > -5%.

Abort:
- Cumulative drift exceeds -5% at any sprint exit without waiver.

## R13 -- Rubric-code drift (NEW, from 2026-04-22 review)

Description: The original 04_evaluation_rubric.md drafted the composite
formula using placeholder field names (`dag_fraction`, `edge_crossings`,
`node_overlaps`, `symmetry_score`, etc.) that do not exist in
`dagua/metrics.py`. Sprints could optimize against a fictional objective.

Mitigation:
- Task 0.8 in Sprint 0 reconciles 04 with code; the revised 04 uses actual
  field names (`dag_consistency`, `crossing_rate`, `overlap_count`, etc.).
- Emergency rubric-change path documented in 04; silent changes prohibited.
- Task 0.8.1 adds `composite_large()` or a fail-loud policy for N>2000 quick
  mode, closing the silent-default blind spot.

Detection:
- 04 references a metric name not in `dagua/metrics.py`.
- Composite at large N silently defaults a required input.

Abort:
- Sprint 0 Task 0.8 must resolve all such mismatches before any sprint exit.

## R14 -- Held-out inspectability (NEW)

Description: Draft plan committed held-out graphs to git and relied on a
"don't iterate on them" convention. Adversarial review: this is security
theater. A clever engineer reads the topologies and tunes to them.

Mitigation:
- Held-out graphs are generated from a SECRET salt at
  `.project-context/private/holdout_salt` (gitignored).
- Only MANIFEST.json (topology hashes, not topologies) is committed.
- Held-out generation is ephemeral: create, metric, destroy. No raw files
  persist on disk between exit runs.
- CI fixture fails if `dagua/graphs/holdout/` contains anything other than
  MANIFEST.json and the `.opaque` marker.
- Rolling set uses the same salt; seed = sha256(salt || sprint_tag)[:8].

Detection:
- Direct `git log` reveals any committed graph tensor in `dagua/graphs/holdout/`.
- pytest fixture failure.

Abort:
- If the salt is compromised (committed accidentally), rotate salt and
  regenerate all held-out metrics. Document rotation in 09 open questions
  followup.

## R15 -- Competitor-parity gap (NEW, from best-in-class goal)

Description: The goal is "best open graph algo in the market." Passing
composite deltas vs our own baseline is not enough if Graphviz dot still
beats us on ResNet-shaped DAGs, or ELK on deep hierarchies, or sgd2_multi
on large undirected graphs. A sprint can exit happy while Dagua is still
dominated by a competitor on a given graph.

Mitigation:
- Per-sprint Pareto gates in 10_iteration_loop.md: sprint N cannot exit
  unless X% of iteration suite is Pareto-optimal vs the 16-variant
  authoritative matrix (11). X is calibrated at Sprint 0.5 baseline and
  ramps to 90% (Sprint 9), with per-family floors enforced.
- Competitor extraction mandatory per sprint (11). "No extraction this
  sprint" is only acceptable with an explicit reason.
- The fourth failed Dagua-native hypothesis on a weak graph FORCES a
  competitor-extraction hypothesis (10_iteration_loop.md north-star rule).
- Runtime is first-class: quality wins at 3x runtime cost do not count.

Detection:
- Pareto share below gate at sprint exit -> exit blocked.
- Runtime drift > 5% per sprint without explicit acceptance -> exit blocked.

Abort:
- Three consecutive sprints with unmet Pareto gate -> pause. Rework
  extraction strategy. If after rework the gate is still unreachable,
  reset expectations with user: maybe 90% is the wrong number, or
  specific families need to be declared out-of-scope.

## R16 -- Extraction without understanding (NEW)

Description: Extracting a competitor technique without understanding WHY it
works tends to produce implementations that copy surface behavior but not
the underlying idea. This is the "cargo-cult" failure mode.

Mitigation:
- Every extraction has a one-line explanation in 11 of WHY the competitor
  technique helps (not just that it exists).
- Every extracted op has a unit test against a hand-computed reference from
  a small graph where the technique's effect is visible.
- Adversarial review post-extraction checks the "why" is preserved.

Detection:
- Extracted op has no unit test, or the test is a trivial round-trip.
- Ablation shows no measurable gain on any family.

Abort:
- Op without test and without ablation gain is removed before sprint exit.

## R17 -- Runtime regression masked by quality gain (NEW)

Description: An extraction that improves quality but makes Dagua 2x slower
shifts the Pareto classification: we were optimal, now we are dominated.
The plan protects quality regression with bars, but runtime is a "stay within
5%" which is softer.

Mitigation:
- 10_iteration_loop.md "runtime is a first-class target": every iteration
  records runtime delta.
- Sprint 9 final gate: iteration-suite average runtime within 1.5x of
  fastest competitor per family.
- Any single iteration that doubles per-graph runtime triggers a mandatory
  review (HJ or adversarial) before keep/revert.

Detection:
- `iteration_log.jsonl` runtime_delta > 50% on any graph without justifier.
- Sprint exit Pareto share unchanged despite quality gains -> check
  if runtime went backward.

Abort:
- Single iteration 2x slower -> revert or justify.

## Meta-risk: Plan rot

If this plan is not revised when a sprint changes reality, it becomes
misleading. Mitigation: every sprint exit note includes a "plan edits" section
listing any file in `.project-context/plans/native_placement_algo/` touched
as a result of that sprint. Adversarial review checks that section against
the diff.
