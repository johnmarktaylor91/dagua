<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **graphopt**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

6 graphopt variants total. 4 partial_match: default (0.146), charge_low (0.143),
mass_low (0.105), spring2 (0.109). 2 weak_equivalent: charge_high (0.148),
mass_high (0.173). 68-75 samples each.

## Source code locations

- Dagua: dagua/layout/ops/init.py (GraphOptInitializePositions, etc.),
  dagua/layout/ops/force.py (GraphOpt* prepare/iter ops),
  dagua/layout/ops/pipelines/graphopt.py
- Reference: /home/jtaylor/projects/_references/igraph/src/layout/graphopt.c
  Plus the COULOMBS_CONSTANT define somewhere in src/layout/

## Prior round work

- ROUND_19_diff_graphopt.md + PROMPT_20_fix_graphopt.md
- R16 attempted init-range [0,1] -> [-1,1] alignment; no measurable effect,
  classified architectural floor.
- Current uncommitted stash had graphopt-related changes; per AUTONOMOUS_STATE
  those were defunct experiments. Verify current source state matches what
  ran in the 100-seed benchmark.
- Possibly need: spring_constant default flip, niter/cool semantics, RNG order
  during iteration, init coords range gating.

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/graphopt/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
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
- WRITE ONLY: the PLAN_*.md file in the specified round_31/graphopt/ dir
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
