# Iteration Loop -- Within-Sprint Quality + Runtime Improvement

This is the CORE work pattern. Every sprint's real activity is the iteration
loop, not the spec-write. Exit criteria are downstream of this loop. Goal:
"best open graph layout library in the market" -- for that, the loop must
treat runtime as a first-class target alongside quality, track Pareto fronts
vs competitors on every graph, and be disciplined about overfit.

## The loop, formally

```
for sprint N:
    start_of_sprint:
        baseline = read(eval_output/native_algo/sprint_<N-1>_exit/metrics.json)
        weak_graphs = pick_weakest(baseline, k=5, criterion="worst_vs_best_competitor")
        random_graphs = rotating_subset(iteration_suite, k=3, seed=sprint_tag)
        log = open(sprint_<N>/iteration_log.jsonl, "a")

    while not converged and iters < 50 and clock < sprint_budget * 0.8:
        hypothesis = diagnose_and_hypothesize(weak_graphs)        # human or Claude
        change = implement(hypothesis)                            # op tweak, new op, etc.
        metrics = run_harness(weak_graphs + random_graphs)        # <=60s
        runtime = measure_runtime(weak_graphs + random_graphs)
        pareto_front = compute_pareto(metrics, runtime, competitors)
        log.write(iter_record)

        if strictly_better_on_all(weak_graphs, no_random_regression):
            keep(change)
        elif mixed:
            split_change_into_focused_ops(change); retry
        else:
            revert(change); note_failed_hypothesis()

    end_of_sprint:
        record full iteration_log, Pareto chart, competitor deltas
        run held-out + rolling set (Pareto, not just composite)
        write sprint_<N>_exit_note.md with the top 3 insights
```

## Iteration journal format

One JSON line per iteration, appended to `sprint_<N>/iteration_log.jsonl`:

```json
{
  "ts": "2026-05-10T14:22:31Z",
  "iter": 17,
  "sprint": 3,
  "composite_profile": "profile_small",
  "metric_version_hash": "a3f8c1...",
  "competitor_cache_version": "2026-05-01T00:00:00Z:aef12",
  "benchmark_suite_version": "iteration_v2:42",
  "hypothesis": "Layer-sweep barycenter before gradient core should reduce crossings on wide_parallel_200",
  "change_kind": "new_op | tweak_op | weight_change | pipeline_reorder | revert",
  "change_detail": "added SugiyamaLayerSweepRefine op at pipeline position 3",
  "weak_graphs": ["wide_parallel_200", "grid_20x10", "random_dag_200"],
  "random_graphs": ["chain_100", "binary_tree_127", "cluster_deep_180"],
  "composite_delta_by_graph": {"wide_parallel_200": +7.3, "grid_20x10": +2.1},
  "runtime_median_ms_delta_by_graph": {"wide_parallel_200": -12, "grid_20x10": +3},
  "runtime_p95_ms_delta_by_graph": {"wide_parallel_200": -10, "grid_20x10": +9},
  "pareto_vs_competitors": {"wide_parallel_200": "now optimal", "grid_20x10": "dominated by graphviz_dot"},
  "decision": "keep | keep_focused | revert",
  "notes": "Sweeping before gradient core works; doing it after makes it worse."
}
```

`composite_profile`, `metric_version_hash`, `competitor_cache_version`,
and `benchmark_suite_version` are REQUIRED per 2026-04-22 adversarial
review (Sprint 9 cannot mine logs across sprints without them). Sprint 0
implements the version-hashing.

`scripts/iter_record.py` (built in Sprint 0) writes one of these records
from a single invocation; it does the harness run, runtime measure, Pareto
compute, and JSON append in one call.

## Runtime is a first-class target

Every sprint has TWO improvement targets, not one:

1. **Quality target**: per-family composite improves vs prior sprint.
2. **Runtime target**: per-graph wall-time either holds within 5% of prior
   sprint OR improves. By Sprint 9 the iteration-suite SAME-DEVICE runtime
   must land per-family-envelope (set at Sprint 0.5):
   - Small DAG / tree (N<=1K): PARITY-ONLY vs graphviz_dot. Dagua CANNOT
     be expected to beat C-based graphviz on small DAGs; target is within
     2x of graphviz_dot (not 1.5x).
   - Medium DAG (1K-20K): within 1.5x of fastest same-device competitor.
   - Large undirected (20K-100K): within 1.5x of fastest same-device
     competitor on GPU.
   - Ultra (100K+): within 2x of sgd2_multi on GPU.
   These envelopes are declared at Sprint 0.5 and become binding gates.
   Device rule: same-device comparison only (per the authoritative
   matrix in 11_competitor_weaving.md).

A sprint cannot exit if its gains in quality come from a 2x runtime cost,
unless the user explicitly accepts the trade (iMessage confirmation).

Runtime measurement rules:
- Warmed: 3 warm-up runs, discard, then 5 measured runs, report median.
- Same device per comparison (CPU-to-CPU, GPU-to-GPU).
- Isolation: no other big processes (pre-flight `free -h`, `nvidia-smi`).
- Record as `runtime_ms_median` and `runtime_ms_p95`.

## Pareto discipline (the hard part)

"Better than competitors" is a Pareto question, not a scalar question. A
layout with +10 quality and +3x runtime may still lose to graphviz dot's
+0 quality and 0.3x runtime for users with large graphs.

Per-graph Pareto rule: dagua is "competitive" on a graph if it is
Pareto-optimal on the (quality, runtime) plane among all benchmarked
competitors. "Pareto-optimal" means no competitor dominates it on BOTH
axes. The Pareto gate is PURE non-domination -- no scalar tie-break.

Reporting (NOT gating) uses the existing
`scripts/quality_runtime_analysis.py` role labels: "balanced", "fastest",
"best-quality", and distance-to-ideal for ranking within the non-dominated
set. These are informational for sprint exit notes; do not redefine
Pareto classification.

Sprint exit Pareto gates are CALIBRATED at Sprint 0.5, NOT fixed in advance.
At Sprint 0.5 exit we measure `baseline_pareto_share_iter` under the frozen
authoritative competitor matrix. The ramp is:

```
gate(N) = baseline + (N / 8) * (target_9 - baseline)
target_9 = 90% iter / 80% held-out
```

Per-family floors ENFORCED at Sprint 9 (prevent a few hard families from
capping the plan; per adversarial review):

| Family | Sprint 9 Pareto floor |
|--------|-----------------------|
| directed DAG (small N<=1K) | >=50% (parity recognized) |
| directed DAG (medium/large) | >=80% |
| tree | >=50% |
| nested cluster | >=85% |
| undirected sparse | >=85% |
| near-clique | >=30% |
| disconnected | >=30% |
| pathological | >=20%, no crash |

The suite-wide target (90% / 80%) AND all family floors must hold at
Sprint 9 exit. This replaces the flat per-sprint ladder flagged as
uncalibrated.

Measured on the iteration suite; held-out share must be within 10% of
iteration share AND per-family held-out Pareto share must be within 10
percentage points of the per-family iteration share (new per round-2
adversarial review: aggregate closeness alone can hide a badly-missed
held-out family).

## Choosing weak graphs (the fuel for iteration)

At sprint start, read baseline metrics and pick the 5 weakest:

Priority for "weak":
1. Graphs where dagua is DOMINATED (another competitor beats both axes).
2. Graphs where dagua has the largest composite gap to best competitor.
3. Graphs where dagua has the largest runtime gap to fastest competitor.
4. Graphs in a priority family (P0) with any regression from Sprint 0.

At most 2 from any one family (force spread). Include at least one
runtime-weak graph and at least one quality-weak graph.

`scripts/pick_weak_graphs.py <sprint_tag> --k=5` (built in Sprint 0).

## Rotating random subset

Per-iteration the harness runs the 5 weak graphs + 3 random graphs sampled
from the iteration suite using `sha256(salt || sprint_tag || iter_number)`.
This forces us to not overfit even to the weak graphs: a change that helps
the weak graphs but breaks a random one is caught early.

Random subset changes every iteration within the sprint, so across ~20
iters per sprint we cover the full iteration suite roughly twice.

## Decision rules (ends of the loop)

- **Keep**: composite delta >= 0 on every weak graph AND no random graph
  regresses > 3%. Commit the change. Next iteration.
- **Keep-focused**: change helps some graphs, hurts others. Split into
  multiple ops (e.g., per-family opt-in via graph classifier) and retest.
- **Revert**: mixed too badly to split, or regressing random graphs. Revert.
  Log the hypothesis as failed. Pick new hypothesis.

Anti-flailing: if 3 consecutive hypotheses on the same weak graph fail,
STOP iterating on that graph. Move to a different weak graph. Document the
stuck graph in the sprint exit note.

## Convergence criterion

A sprint's iteration loop exits when ANY:
1. 3 consecutive iterations yield `keep` with composite delta < 0.5 AND
   runtime mean delta < 2% AND runtime p95 delta < 2% AND no pathological-
   family tail improvement > 3%. (true plateau on both central tendency
   and tail)
2. Weak graphs' composite has improved >= sprint's quality target.
3. Clock budget exhausted (80% of declared sprint budget).

A `tail-improved keep` path (new per adversarial review): if an iteration
does NOT improve composite mean on any weak graph BUT reduces p95 runtime
on any pathological graph by >5% OR reduces worst-case composite on any
family by >3%, classify as `keep_tail_improved` rather than noise.
Do not reset the plateau counter for these.

Then run the full iteration suite + held-out + rolling + competitor head-to-head
for the sprint exit metrics.

## Competitor integration trigger

If the loop stalls on a weak graph (3 failed hypotheses), mandatory next
step: look at the best competitor on that graph and extract one technique
per 11_competitor_weaving.md. "I tried 3 things, none worked" is not an
acceptable sprint exit state while competitors still dominate that graph.

## Iteration logs drive Sprint 9

Sprint 9 tuning is data-driven off accumulated iteration_log.jsonl files,
not a clean-slate Optuna run. Claude reads all eight sprints of logs,
identifies consistent winners (changes that kept across sprints) vs
one-offs, and writes the final weight tuning spec based on evidence.

## What this loop PROTECTS AGAINST

- **Overfit**: rotating random subset + held-out gate.
- **Runtime drift**: explicit runtime target per sprint.
- **Invisible regressions**: per-iteration composite + runtime delta logged.
- **"Should be better" claims without proof**: Pareto gate is a formula,
  not a vibe.
- **Flailing**: 3-fail rule on a weak graph forces a different approach.
- **Lost insights**: iteration log survives sprints; Sprint 9 reads them.

## What it does NOT protect against

- Aesthetic drift hidden by composite gains. Mitigation: HJ rotation per 04.
- Competitor benchmark staleness: see 11 for the refresh policy.
- Plan rot in 10 and 11 themselves: revised when a sprint reveals a gap.
