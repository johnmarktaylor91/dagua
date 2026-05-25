<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **lgl**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

All 5 lgl variants: **weak_equivalent**. Median RMSD 0.13-0.15 vs igraph_lgl.
Variants: lgl_cool1 (0.140), cool2 (0.136), default (0.139), iter300 (0.132),
iter50 (0.146). 93-95 samples each. These PASS equivalence but at looser margin;
tightening to strong_equivalent is the goal.

## Source code locations

- Dagua: dagua/layout/ops/pipelines/lgl.py
- Reference: /home/jtaylor/projects/_references/igraph/src/layout/large.c
  (LGL = Large Graph Layout; Adai et al. 2004)

## Prior round work

- ROUND_21_DIFF_lgl.md identified: 'igraph IGNORES edge weights, dagua applies them'.
- R22 commit 3a44668 + R23 commits beb10ff/93f3199: 'align weights and convergence'
  + validation warnings + summary.
- Current weak_equivalent means TOST passes but at the looser margin. Specific
  divergences to investigate: cooling schedule per-iteration semantics, energy
  function constants, initial layout (random vs degree-ordered), reordering
  vs not reordering each iteration.

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/lgl/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
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
- WRITE ONLY: the PLAN_*.md file in the specified round_31/lgl/ dir
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
