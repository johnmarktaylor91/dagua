# Evaluation Rubric

## Source of truth

The composite formula is defined in `dagua/metrics.py:composite` (lines
1147-1206). This plan file DOCUMENTS that code. If code and this file disagree,
CODE WINS until Sprint 0 Task 0.8 reconciles them and re-freezes both.

Revised per adversarial review 2026-04-22: the original rubric used
rubric-land metric names (`dag_fraction`, `edge_crossings`, `node_overlaps`,
`symmetry_score`, `cluster_violations`, `label_clipping`, `bbox_outside_frame`)
that do not exist in `dagua.metrics`. That mismatch is corrected below.

## Actual composite (verified against code)

```
# code: dagua/metrics.py:1147
score = 0
score += 25 * dag_consistency                             # main DAG quality
score += 20 * max(0, 1 - edge_length_cv)                  # edge-length uniformity
score += 15 * max(0, depth_spearman_rho)                  # depth-vs-y correlation
score += 10 * (1 if overlap_count == 0 else 0)            # hard no-overlap (binary)
score += 10 * max(0, 1 - edge_straightness_mean_deg/45)   # edges near vertical
score += 10 * max(0, 1 - crossing_rate * 10)              # crossing density
score +=  5 * min(1, angular_res_mean_deg / 40)           # angular resolution
score +=  5 * (min(1, cluster_mean_sep_ratio/5) if "cluster_mean_sep_ratio" in metrics else 0.5)
# Optional fields (only added if present in metrics dict):
if "edge_node_crossing_rate" in metrics:
    score +=  3 * max(0, 1 - edge_node_crossing_rate * 5)
if "label_overlaps" in metrics or "label_node_overlaps" in metrics:
    total_label_overlaps = label_overlaps + label_node_overlaps
    lo_score = 1.0 if total_label_overlaps == 0 else max(0, 1 - total_label_overlaps * 0.1)
    score +=  2 * lo_score
# Max theoretical: 105. Practical cap: ~100 depending on graph features.
```

## Missing metrics (flagged by adversarial review)

The meta-prompt mentions aesthetic criteria that are NOT currently implemented:
symmetry, cluster containment violations, label clipping, bbox-outside-frame,
angular-resolution-penalty beyond the existing `angular_res_mean_deg`. Sprint 0
Task 0.8 reconciles: either implement the missing metrics (if a sprint will
use them) or remove them from plan language. Until reconciled, the composite
is the one above, nothing more.

## Scoring behavior at scale (formerly a blind spot)

`dagua/metrics.quick()` (lines 1214-1293) runs a reduced Tier-1 set; `full()`
(1299-1398) adds crossing rate, angular resolution, cluster separation,
edge-node crossings, label overlaps. `dagua/eval/benchmark.py:835` cuts over
to quick mode at N>2000. That means at N>2000:

- `crossing_rate` is absent; code defaults to 0.5, which contributes 0.0 to
  the composite (max(0, 1 - 0.5*10) = 0).
- `angular_res_mean_deg` defaults to 20, contributing 2.5 (half of full 5).
- `cluster_mean_sep_ratio` absent; contributes neutral 2.5.
- `edge_node_crossing_rate`, `label_overlaps` absent; drops up to 5 pts.

Total silent drop at large N: up to ~15 points vs small-N runs even on
identical topology quality. This produces misleading cross-tier comparisons.

**Policy (revised)**: composite scores are ONLY comparable within the same
"composite profile":
- `profile_small` = N<=2000 with full metrics; max 105.
- `profile_large` = N>2000 quick metrics; max ~90; crossings/angular/cluster
  treated as "unknown" not "default 0.5".
- Sprint exits compare composite within-profile only. Cross-tier claims must
  quote per-profile scores separately.

Sprint 0 Task 0.8 implements either (a) a distinct `composite_large()` that
uses quick-available fields and renormalizes to 0-100, or (b) an assertion
that fails the run if composite inputs are missing. Preference: (a) with a
flag in metadata so dashboards never mix profiles silently.

## Qualitative rubric (aesthetic, adversarial-agent + HJ)

Per-graph scored on 5-point scale by a review pair:

| Criterion | 1 = bad | 5 = good |
|-----------|---------|----------|
| Visual clarity | Illegible | Immediately readable |
| Flow | No sense of direction | Unambiguous flow |
| Aesthetic balance | Crowded or sparse | Balanced whitespace |
| Cluster coherence | Hierarchy invisible | Hierarchy clear |
| Label readability | Clipped or overlapping | All labels visible |

Reviewer pair:
- Reviewer A: `/codex:review` (focus="aesthetic") with prompt template A.
- Reviewer B: Claude subagent (general-purpose) with prompt template B that
  uses DIFFERENT framing (per 06 adversarial protocol) to reduce circularity.
- Disagreement > 1 point triggers a fresh run with a third prompt variant;
  persistent disagreement escalates to HJ.

Known-circularity risk: both reviewers are LLMs. Mitigations in 06.

## Human judgment (HJ) -- iMessage protocol

Triggers:
1. Sprint exit: mandatory 3x3 grid (one per sprint).
2. Adversarial disagreement > 1 point on any flagship or non-flagship graph.
3. Quality regression > 5% on a P0 family vs prior sprint's held-out.
4. **NEW** Blind spot-check: every two sprints, one grid from RANDOM
   non-flagship held-out graphs across families (not curated). Catches slow
   drift on underweighted families.
5. **NEW** Two consecutive negative quality deltas on any family (even if
   individually < 3%): HJ ping with that family's delta history.

Format: `send-to-jmt.sh -a <grid.png> "<caption>"`. Grid: rows = 3 graphs,
cols = sprint N-1 | sprint N | best-seen. Caption names which trigger fired.

User silence > 6 hours = tentative approval, sprint marked "unverified by
user."

## Competitor-delta scoring (new, from "best in class" goal)

Raw composite is insufficient for "best open graph algo." Dagua must be
Pareto-optimal vs competitors. For each graph, compute:

```
competitor_delta_composite = dagua_composite - max(competitor_composite)
competitor_delta_runtime   = fastest_competitor_runtime / dagua_runtime
pareto_class = {"optimal", "dominated", "tied"} per 10_iteration_loop.md
```

Sprint exit requires the per-sprint Pareto gate from 10 and the aggregate
`competitor_delta_composite >= 0` on >= the gated fraction of iteration
suite. See 11 for competitor list.

Rollup per sprint exit:
- `pareto_share_iter` = fraction of iteration suite where Dagua is Pareto
  optimal.
- `pareto_share_holdout` = same, on held-out.
- `|pareto_share_iter - pareto_share_holdout| < 10%` (anti-overfit gap).

## Composite ranking + family vetoes (revised)

Per-graph composite averaged per family then weighted:

| Family | Weight | Veto bar (hard) |
|--------|--------|-----------------|
| directed_dag | 2.0 | composite -3% cap per sprint |
| tree | 1.0 | composite -3% cap |
| nested_cluster | 1.5 | composite -5% cap |
| undirected_sparse | 1.0 | composite -5% cap |
| near_clique | 0.5 | **composite -10% veto** |
| disconnected | 0.5 | **composite -10% veto** |
| pathological | 0.2 | **composite -15% veto + hard constraint no crash** |

"Veto" means: the sprint is NOT exited if that family's composite degrades
past the bar, even if the weighted total is OK. This prevents a gain on DAGs
from hiding a catastrophic regression on disconnected or pathological graphs.

## Anti-overfit gating (revised, enforceable)

Formulas, not prose:

```
overfit_gap        = mean(iter_composite) - mean(holdout_composite)
rolling_gap        = mean(holdout_composite) - mean(rolling_composite)
cumulative_drift   = mean(current_holdout_composite) - mean(sprint_0_holdout_composite)
family_cumulative  = per-family cumulative drift, same formula scoped
```

Sprint exit bars (additions):
- `overfit_gap < 10%` relative, measured per-profile.
- `rolling_gap < 15%` relative, measured per-profile.
- `cumulative_drift > -5%` absolute on composite; breaching this requires
  user waiver before the sprint exits.
- Every `family_cumulative > (family veto bar)` is an exit blocker.

All four gates are computed in `scripts/quality_runtime_analysis.py`
(extended in Sprint 0 Task 0.8.2). Missing any numerator/denominator = exit
fail; no silent defaults.

## Freeze protocol + emergency change path (revised)

Composite coefficients are frozen after Sprint 0 Task 0.8 syncs them with
code. Sprints 1-8 may NOT tune them except via the emergency path:

Emergency path requirements:
1. Claude writes an emergency change note in the current sprint file with:
   old coefficient, new coefficient, explicit rationale, list of prior sprint
   results that must be re-scored.
2. User approves the change via iMessage reply (not silent timeout).
3. All prior sprint-exit metrics files are re-computed under new weights and
   stored as `metrics_recomputed.json` alongside originals.
4. The emergency change is logged in `08_risk_register.md`.
5. Rollback if re-scoring reveals the change introduces new regressions:
   revert coefficient, document in the sprint note.

Any mid-plan rubric change that skips the emergency path is a CRITICAL
process violation and triggers a retrospective.

## Metric pitfalls (retained from draft)

| Metric | Pitfall | Mitigation |
|--------|---------|-----------|
| crossing_rate | Absent at N>2000 quick mode | Use composite_large profile |
| dag_consistency | Undefined for undirected | Skip composite term for undirected |
| edge_length_cv | Rewards artificially-uniform layouts | Already bounded 0-1 |
| depth_spearman_rho | Undefined for undirected | Skip for undirected |
| angular_res_mean_deg | O(sum d^2); slow on hubs | Sample hub neighborhoods |
| cluster_mean_sep_ratio | Default 0.5 when no clusters | Use min-only with separate flag |

## Reviewer diversity (from adversarial finding)

Both LLM reviewers share a model family, creating rubric circularity.
Mitigation ladder (applied until disagreement rate drops below 20%):

- Level 1: Use different agent harnesses for Reviewer A and B (already: Codex
  vs Claude subagent).
- Level 2: Reviewer prompt templates explicitly diverge. A = "attack this
  layout as if you were a paying user who hates it"; B = "rate this layout
  against Purchase 1997 with citations".
- Level 3: If L1+L2 still circular, add a Level-3 review by the user
  (HJ) on a stratified sample.

## What we are NOT scoring

- Absolute runtime. Separate budget per tier (03_test_matrix.md).
- Memory usage. Separate budget (scaling principles).
- Edge label placement quality. Separate visual audit until Sprint 7.
- Theme / style. Rendering is out of scope for this plan.
