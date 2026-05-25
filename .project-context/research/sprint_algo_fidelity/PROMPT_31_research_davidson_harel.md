<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **davidson_harel**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

davidson_harel: rounds50 (0.194, partial, 8 samples), rounds100 (0.168, partial,
3 samples), rounds200 (insufficient_data, 0 samples). Reference: igraph_davidson_harel.

## Source code locations

- Dagua: dagua/layout/ops/davidson_harel.py + dagua/layout/ops/pipelines/davidson_harel.py
- Reference: /home/jtaylor/projects/_references/igraph/src/layout/davidson_harel.c

## Prior round work

- Multiple rounds of work: R12 BLOCKED (timeout), R13 LANDED (0.362 -> 0.238,
  divergent->partial; commit 0fac3e5), R20 LANDED (-0.071 median, equivalent_at_1x;
  commit e58728f -- fine-tuning phase + delta-energy + skip_finalization).
- Despite that work, rounds50/100 only get 3-8 samples and rounds200 gets 0.
  This is mostly the watchdog-stuck cascade for slow stochastic engines on dense
  graphs. So the variant DOES converge on graphs it completes; it just can't
  complete on most graphs at scale.
- Lever: tighten the inner per-layout 600s timeout to e.g. 120s for known-slow
  engines, so they trigger clean per-layout timeouts instead of cascading to
  pool-stuck. OR: identify what specifically hangs and add an internal early-exit.

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/davidson_harel/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
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
- WRITE ONLY: the PLAN_*.md file in the specified round_31/davidson_harel/ dir
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
