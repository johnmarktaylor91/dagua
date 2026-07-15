<task>
Implement the APPROVED metrics/criteria fixes from dagua's r75 adversarial critique (verdicts 30,
31, 35 in .project-context/research/sprint_rng_matching/r75_findings/r75_ADVERSARIAL_VERDICTS.md
-- read it first, plus r75_metrics_codex.md and r75_metrics_sonnet.md for full context).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-metrics-fixes (branch r75/metrics-fixes
@ 89ed3c3). Work ONLY here. Conventional commits (fix(eval): ...). No push, no merge.

THREE CHANGES (and NOTHING else -- margin widening was REJECTED, do not touch margins):

1. SAMPLED-CROSSINGS ESTIMATOR (verdict 30, gate: cross_sampled=True rows only; exact rows must
   not move):
   - dagua/metrics.py sampled_crossing_rate (~:790-832): fix the denominator bug -- the valid-pair
     conditional rate is currently scaled by ALL E*(E-1)/2 unordered pairs including ineligible
     adjacent pairs (:831). Estimate the eligible (non-adjacent) pair count and scale by that, or
     estimate totals via the eligible-pair sample directly.
   - Propagate the standard error: return (estimate, se, n_valid, eligible_pairs) to the fidelity
     path; scripts/definitive_fidelity_analysis.py crossing_count (~:1726-1745) currently discards
     crossing_se -- persist it in the row (new fields cross_se_D/cross_se_R) and fold it into the
     SAMPLED-path TOST margin as max(existing_margin, z*sqrt(se_D^2+se_R^2)) with z=1.96 ONLY when
     cross_sampled is True. Exact-count rows: zero behavioral change (add a regression test
     asserting so on an E<=500 fixture).
2. EXACT/SAMPLED PREDICATE CONSISTENCY (verdict 31): the exact path (_segments_intersect_scalar,
   metrics.py:2068-2085) and the vector path (segments_intersect, :146-201) disagree on collinear
   overlap. Unify: use the vector predicate semantics (documented degeneracy behavior) in the
   exact batch path. Add regression tests: collinear-overlap pair, endpoint-touch pair,
   near-parallel non-overlap pair, and the same graph evaluated at E=500 vs E=501 must agree.
3. QUALITY_SUPERIOR_DISTINCT metadata tier (verdict 35): in scripts/definitive_fidelity_analysis.py
   add a persisted boolean field quality_superior_distinct: True when the combo FAILS the
   two-sided battery but dagua is strictly better on every failing leg (stress/cross lower,
   np higher, each beyond its margin). This is TRIAGE METADATA ONLY: it must NOT feed
   quality_identical_raw, the battery tier, rungs, or any headline count. Mirror the existing
   np one-sided pattern (:1642-1669) for direction logic. Add a synthetic-fixture test: a combo
   where dagua is strictly better on all three -> flag True, quality_identical_raw stays False.

VERIFICATION LOOP:
a. pytest tests/test_quality_battery_correctness.py tests/ -k "crossing or battery or quality" -x -q
   -- all green including your new tests. NEVER weaken an existing assertion.
b. REPLAY CHECK (the critique requires this): re-run the battery decision logic over the existing
   409-row eval_output/fidelity_definitive/r74_phase2_rescore.jsonl fields (read-only from the
   MAIN repo /home/jtaylor/projects/dagua/eval_output) where possible, or a representative
   recomputation on 10 exact-count combos, demonstrating: exact-count rows produce IDENTICAL
   pass/fail decisions before/after. Any changed exact-row decision = your change is wrong.
c. Controls: run python3 scripts/definitive_fidelity_report.py --controls --controls-dir
   eval_output/fidelity_definitive/controls --output-dir /tmp/r75_metrics_controls (copy the
   controls dir into the worktree or point at the main repo path read-only) and confirm gate_5
   remains 0/40. If the controls harness cannot run standalone, document exactly why and provide
   the synthetic-fixture evidence instead.
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_metrics_NOTES.md:
what changed, commits, replay evidence, controls result.
</task>
<completeness_contract>
Done = 3 changes implemented + gated + tested, replay shows exact rows unmoved, controls/fixtures
green, notes written. Do NOT implement the no-canonical-reference tier (awaits JMT sign-off), the
huge-graph approximate path (separate task), or any margin floor changes (REJECTED).
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/metrics-fixes only. Do not touch layout pipeline code.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
