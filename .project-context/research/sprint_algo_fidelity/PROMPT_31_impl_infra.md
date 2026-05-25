<task>
Round 31 IMPLEMENTATION for infra recovery (biggest sample-recovery lever).

Read first:
- eval_output/algo_fidelity/round_31/infra_recovery/PLAN_claude.md
- eval_output/algo_fidelity/round_31/infra_recovery/PLAN_jtaylor_zmachine_20260524_174638.md (if present)
- eval_output/algo_fidelity/round_31/davidson_harel/PLAN_claude.md (same root cause)
- eval_output/algo_fidelity/round_31/ROUND_31_INTEGRATED_PLAN.md (A5)

## Implement (in order, each as separate commit)

### I1: Scope watchdog to single stuck future (BIGGEST LEVER, ~50 LoC)
`scripts/run_benchmark.py:2336-2386` -- when ONE future stalls past WATCHDOG_TIMEOUT (7200s), the entire inflight window of ~200 futures gets marked `error: watchdog: worker pool stuck` and pool is rebuilt. This cascade is the root cause of 109,807 stuck rows in the 100-seed run.

Fix: track per-future start time; cancel ONLY futures that exceed individual timeout; do not poison healthy peers.

### I2: Drop default WATCHDOG_TIMEOUT 7200 -> 600
`scripts/run_benchmark.py:85` (or wherever WATCHDOG_TIMEOUT default lives).

### I3: Per-engine timeout caps in `_BASE_TIMEOUT_CAPS`
`dagua/eval/variants.py:1945-1948`:
```python
_BASE_TIMEOUT_CAPS["classic_neulay"] = 180
_BASE_TIMEOUT_CAPS["classic_sgd2_multi"] = 120
_BASE_TIMEOUT_CAPS["classic_davidson_harel"] = 180  # was 60
```

### I4: Per-engine max_nodes caps in variant entries (dagua/eval/variants.py)
- neulay variants: max_nodes=1500
- sgd2_multi_with_crossing: max_nodes=500
- other sgd2_multi: max_nodes=2000
- davidson_harel variants: max_nodes=300

### I5: NeuLay autograd-related safeguards (already partially in R30 commit 07b6d62)
Verify R30 fix is active. Add NaN guard after optimizer.step() (dagua/layout/ops/neulay.py).

### I6 (if reference adapters missing): document
infra_recovery codex flagged: NeuLay reference not importable as `neulay`/`NeuLay`; sgd2_multi reference may be missing. If so, add tracking comments to dagua/eval/competitors/*; do not silently produce 0 paired rows.

## Verification

After I1-I4 land, run focal benchmark on a small subset to confirm no cascades:
```bash
python scripts/run_benchmark.py --seeds 5 --variants \
    --engines classic_neulay_default,classic_sgd2_multi_default,classic_davidson_harel_rounds50 \
    --graphs ba_500,small_world_500 \
    --output-dir /tmp/r31_infra_smoke \
    --timeout 300 --watchdog-timeout 1200
```

Expected: no "worker pool stuck" cascades; clean per-layout timeouts only.

## Scope
- DO NOT TOUCH: render/styles, cluster sprint, existing fidelity_report/benchmark_100seed_final outputs
- Explicit git add. Commit: `fix(bench): round 31 infra -- <terse>`.
- Multiple commits OK.

## Output
`eval_output/algo_fidelity/round_31/infra_recovery/SUMMARY.md` documenting changes + smoketest result.
</task>

<completeness_contract>
I1 + I2 + I3 + I4 minimum. I5/I6 if cheap.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
