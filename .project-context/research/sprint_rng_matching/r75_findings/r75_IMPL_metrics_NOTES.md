# r75 metrics fixes implementation notes

Date: 2026-07-01
Branch: `r75/metrics-fixes`

## What changed

- `dagua/metrics.py`
  - `sampled_crossing_rate()` now scales the eligible-pair conditional crossing rate by the
    eligible non-adjacent edge-pair population, not by all `E choose 2` unordered edge pairs.
  - `crossing_se` is now in crossing-count units over the eligible-pair total.
  - Exact `count_crossings()` now batches through `segments_intersect()` so exact and sampled paths
    share the documented collinear-overlap and endpoint-touch semantics.
- `scripts/definitive_fidelity_analysis.py`
  - Added `CrossingEstimate` and `crossing_estimate()` for fidelity rows.
  - Persisted sampled crossing diagnostics: `cross_se_D`, `cross_se_R`, `cross_n_valid_D`,
    `cross_n_valid_R`, `cross_eligible_pairs_D`, `cross_eligible_pairs_R`.
  - `quality_cross_margin()` adds `1.96 * sqrt(se_D^2 + se_R^2)` only when `cross_sampled=True`.
    Exact-count rows keep the old margin behavior.
  - Added `quality_superior_distinct` metadata for non-identical rows where every failed battery
    leg favors Dagua beyond the relevant margin. It is not used by `quality_identical_raw`,
    `q_battery`, final rungs, or headline counts.
- Tests
  - Added crossing predicate regressions for collinear overlap, endpoint touch, near-parallel
    non-overlap, and 500/501-edge predicate consistency.
  - Added sampled denominator regression with many adjacent edge pairs.
  - Added exact-row sampled-SE no-op regression and synthetic `quality_superior_distinct` fixture.

## Commits

- No commit made. The task text asked for conventional commits, but the worktree instructions say
  the orchestrator handles git operations and not to commit or push.

## Replay evidence

Field replay over read-only main-repo file:
`/home/jtaylor/projects/dagua/eval_output/fidelity_definitive/r74_phase2_rescore.jsonl`

Output:

```text
rows=409 exact_rows=409 sampled_rows=0
exact_decision_changes=0
```

Interpretation: all existing r74 Phase-2 rows are exact-count rows, and replaying battery
pass/fail decisions from persisted fields produced identical exact-row decisions before/after.

## Controls result

Command:

```bash
python3 scripts/definitive_fidelity_report.py --controls \
  --controls-dir /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/controls \
  --output-dir /tmp/r75_metrics_controls
```

The controls harness ran standalone but exited nonzero because not all gates pass. Requested gate:

```text
gate_5_quality_identical_laundering passed=True scored=40 three_q_count=0
```

Other gate status from `/tmp/r75_metrics_controls/controls/gate_results.json`:

```text
gate_1_positive_mode_a passed=True scored=39 pass_count=39
gate_2_positive_mode_b passed=True pass_count=39
gate_3_negative passed=False scored=20 non_primary_percent=90.0
gate_4_chance passed=True ks_p=0.8177374448453805
gate_5_quality_identical_laundering passed=True scored=40 three_q_count=0
gate_6_reference_self_split_positive passed=False scored=0 three_q_count=0
all_passed=False
```

## Verification

Passed:

```text
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_quality_battery_correctness.py tests/ -k "crossing or battery or quality" -x -q
# 131 passed, 3021 deselected, 43 warnings
pytest tests/test_metrics.py tests/test_metric_seeding.py tests/test_quality_battery_correctness.py -x --tb=short -q
# 54 passed, 3 warnings
pytest tests/test_graph.py -x --tb=short -q
# 37 passed, 3 warnings
```

Attempted but not clean:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

This printed passing progress for about 24 minutes, then the session ended with code `-1` and no
pytest failure summary.

Final non-slow suite:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

Stopped on an unrelated existing benchmark checkpoint test:

```text
FAILED tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest
AssertionError: assert _load_hierarchy_checkpoint(...) is None
```

The isolated rerun of that single test also fails the same way. I did not modify benchmark
checkpoint code because it is outside the approved metrics/criteria scope.

## Assumptions and choices

- Sampling SE is stored as the standard error of each crossing-count estimate. The battery margin
  uses the pooled SE of the mean D/R crossing difference, which is conservative because the sampled
  D/R pair set is shared by seed.
- `quality_superior_distinct` requires the row to remain non-identical and every failed leg to be
  better beyond margin. Passing legs do not have to fail symmetrically; this preserves the existing
  one-sided NP policy.
- `_segments_intersect_scalar()` is now unreachable from `count_crossings()` after the exact path
  was unified on `segments_intersect()`. It is removable in a later cleanup if no external imports
  depend on it.

## Out of scope

- No margin floor widening.
- No no-canonical-reference tier.
- No huge-graph approximate scoring path.
- No layout pipeline changes.
