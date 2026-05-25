<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **infra_recovery**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

INFRASTRUCTURE TARGET: recover 14 'insufficient_data' variants -- all 6 neulay
variants, all 8 sgd2_multi variants. 0 samples each in the 100-seed run.

R30 dagua-internal fixes already committed (POST-benchmark, not yet measured):
- 07b6d62: neulay/tsnet enable_grad() fix
- 5168b9d: dagua native CUDA OOM detection + CPU fallback

The 100-seed benchmark recorded 109k 'watchdog: worker pool stuck' errors from
~549 watchdog recycle events. These engines were the worst offenders.

## Source code locations

- neulay: dagua/layout/ops/pipelines/neulay.py, dagua/layout/ops/neulay.py
- sgd2_multi: dagua/layout/ops/pipelines/sgd2_multi.py + sgd2_multi.py if exists
- Worker pool implementation: scripts/run_benchmark.py (look for watchdog timeout,
  worker pool stuck, executor recycling logic)

## Prior round work

- Engines with FULL cascade (all 100 seeds stuck across many graphs):
  classic_neulay_default/lr001/lr05/no_gcn/radius02/radius08 on 10 graphs each.
  classic_sgd2_multi_default/with_crossing/lr001/lr01/stress_only/with_aspect/batch128/batch8.

- For infra: tightening --timeout from 600s -> 120s for these engines might
  convert watchdog cascades into clean per-layout timeouts (fewer stuck-cascade
  rows). Alternatively: per-engine max_nodes caps to skip large graphs entirely.

- For algo: even with infra fixed, do the engines actually converge on the
  graphs where they DON'T hang? Quick check via algo_fidelity_live_compare on
  bounded 5-graph subset will reveal whether they have algorithmic divergences
  too.

Research: identify (a) what specifically hangs, (b) what timeout/skip strategy
recovers most variants, (c) whether post-R30 these engines work cleanly on
the bounded subset.

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/infra_recovery/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
(or any unique filename in that dir) with:

1. **Root-cause analysis per variant** in this family that isn't strong_equivalent.
   For each: identify specific algorithmic divergences from the reference, with
   file:line on BOTH dagua and reference sides.

2. **Ranked fix list**. Each item has:
   - Concrete description
   - Estimated lines of code changed (net)
   - Risk (low/medium/high)
   - Expected RMSD delta if applied (your honest guess)
   - Implementation sketch (pseudocode where useful)

3. **Stop conditions**: items you believe CAN'T be fixed without invasive
   reference-side patches. But the user explicitly said "NOTHING deferred" --
   so if it's invasive but possible, include it with cost estimate. Only mark
   as "truly cannot be fixed" if it's a hard architectural mismatch.

Read the reference source line by line where useful.

## Scope

- DO NOT TOUCH: any dagua/* source files
- DO NOT TOUCH: render/styles, cluster sprint files
- DO NOT TOUCH: existing eval_output/fidelity_report_100seed_final/* outputs
- DO NOT TOUCH: existing eval_output/benchmark_100seed_final/* outputs
- WRITE ONLY: the PLAN_*.md file in the specified round_31/infra_recovery/ dir
- Output as markdown, target 200-600 lines

## Adversarial framing

Your rival on this same target is a competing-lab model. We are going to compare
your plans and either pick the better one or synthesize. The rival is good --
make yours BETTER by being more thorough, more specific, more correct. Quote
reference source where they prove your point.
</task>

<research_mode>
Diagnostic round only. Output is the PLAN_*.md file.
</research_mode>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Read deeply.
</default_follow_through_policy>
