# Sprint-39 No-Fix Report

Date: 2026-04-26
Branch: codex/sprint-31a-gate-refinement

## Outcome

No fix shipped. I implemented a class-gated `_post_gradient_barycenter_snap`
candidate in `_best_of_polish`, then ran `/tmp/sprint39_validate.py`. The
candidate did not improve deterministic target crossing rate on any target
graph, and the jitter validation found a large regression on
`dependency_graph_100`. I reverted the code change and made no commit.

## Validation Output

```text
Sprint-39 validation: baseline snap-disabled vs snap-enabled

Targets
dependency_graph_100             composite   59.706 ->   59.706 delta  +0.000 | crossing 0.1160 -> 0.1160 delta +0.0000
                                 crossing rank lift +0.0
mixed_width_labels               composite   87.136 ->   87.136 delta  +0.000 | crossing 0.0000 -> 0.0000 delta +0.0000
                                 crossing rank lift +0.0
random_dag_200                   composite   74.442 ->   74.442 delta  +0.000 | crossing 0.0575 -> 0.0575 delta +0.0000
asymmetric_hourglass_hub         composite   84.836 ->   84.836 delta  +0.000 | crossing 0.0125 -> 0.0125 delta +0.0000
                                 crossing rank lift +0.0
long_range_residual_ladder       composite   82.117 ->   82.117 delta  +0.000 | crossing 0.0223 -> 0.0223 delta +0.0000
                                 crossing rank lift +0.0

Protected
deep_chain_20                    composite   97.500 ->   97.500 delta  +0.000 | crossing 0.0000 -> 0.0000 delta +0.0000
random_dag_200                   composite   74.442 ->   74.442 delta  +0.000 | crossing 0.0575 -> 0.0575 delta +0.0000
ba_500                           composite   63.138 ->   63.138 delta  +0.000 | crossing 0.1209 -> 0.1209 delta +0.0000
org_chart_deep                   composite   92.830 ->   92.830 delta  +0.000 | crossing 0.0000 -> 0.0000 delta +0.0000
hub_fanout_label_skew            composite   93.737 ->   93.737 delta  +0.000 | crossing 0.0000 -> 0.0000 delta +0.0000

Jitter sigma=0.5, 8 trials
dependency_graph_100             crossing lift mean -0.0030, min -0.0798, trials -0.010, -0.003, +0.049, +0.001, +0.002, +0.001, +0.016, -0.080
random_dag_200                   crossing lift mean +0.0001, min -0.0080, trials -0.003, +0.027, -0.003, -0.005, +0.002, -0.008, -0.002, -0.006
long_range_residual_ladder       crossing lift mean +0.0000, min +0.0000, trials +0.000, +0.000, +0.000, +0.000, +0.000, +0.000, +0.000, +0.000

PASS conditions
target_crossing             : FAIL
target_composite            : PASS
protected_composite         : PASS
jitter_no_large_regression  : FAIL

SUMMARY: FAIL
```

## Diagnosis

The deterministic target runs were all exactly unchanged, which means the
candidate either scored below the current picker winner or collapsed back to an
equivalent layout. The jitter run shows the mechanism is not harmless when it
does fire: one `dependency_graph_100` trial regressed crossing rate by `0.0798`.

This is not the sprint-37/37b loss-term failure mode. The candidate generation
path worked, but the generated candidate did not satisfy the validation gate.

## Files

- Reverted: `dagua/layout/ops/pipelines/dagua_native.py`
- Created: `/tmp/sprint39_validate.py`
- Added report: `SPRINT_39_NO_FIX.md`

## Follow-Up

The post-gradient barycenter snap should not ship in this form. A safer future
attempt would need a stronger layer reconstruction guard and a pre-score
crossing check before the composite picker sees the candidate.
