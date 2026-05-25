<task>
R32 minor tightening (fa2_dissuade_hubs + stress_sgd).

Read first:
- eval_output/algo_fidelity/round_31/minor_tightening/PLAN_claude.md
- eval_output/algo_fidelity/round_31/minor_tightening/PLAN_jtaylor_zmachine_20260524_174645.md

## fa2_dissuade_hubs (measurement fix, ~5 LoC)

Claude finding: `outboundAttractionDistribution=True` IS the "dissuade hubs" feature in `fa2_modified` (per `forceatlas2.py:52` comment). Dagua treats them as TWO independent knobs and divides attraction by source mass TWICE when both are True (`dagua/layout/ops/force.py:1947-1950` or similar).

Fix: when `outboundAttractionDistribution` is True, ignore the redundant `dissuade_hubs` divide. Or alternatively make `dissuade_hubs=True` a no-op when `outboundAttractionDistribution=True`.

Verify: classic_fa2 strong_equivalent for 10 of 11 variants; this is the holdout at RMSD 0.104.

## stress_sgd tightening

Currently weak_equivalent at RMSD 0.04-0.05 (eps001, eps01, steps30, steps300). Tightening targets:
1. **Eps semantics**: ensure eps001/eps01 thresholds match reference's stopping criterion exactly
2. **Learning rate schedule**: check if dagua uses the same eta schedule as s_gd2 reference (look at `_schedule_from_weights` vs s_gd2's `default_schedule`)
3. **Batch construction order**: per-epoch pair shuffle order. s_gd2 uses C++ RandomKit/std::mt19937; dagua uses numpy shuffle. Same seed != same draws. Hard architectural floor BUT verify dagua isn't ALSO using a different shuffle bound.
4. **Max steps**: ensure `steps300` etc. forwards correctly through the pipeline.

Apply ONLY items where you can find a concrete bug, not the RNG bridge (architectural).

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_fa2 fa2_ref --seeds 30 --graphs <bounded-5-graph-list> --output-dir eval_output/algo_fidelity/round_32/fa2/post_impl
python scripts/algo_fidelity_live_compare.py classic_stress_sgd sgd2 --seeds 30 --graphs <bounded-5-graph-list> --output-dir eval_output/algo_fidelity/round_32/stress_sgd/post_impl
```

## Scope
- DO NOT TOUCH: render/styles, cluster sprint, existing benchmark outputs
- Explicit git add. Small commits.
- Commits: `fix(layout): round 32 fa2 -- <terse>` and `fix(layout): round 32 stress_sgd -- <terse>`.

## Output
SUMMARY per target.
</task>

<completeness_contract>
At least fa2 fix committed. stress_sgd if real bugs found.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
