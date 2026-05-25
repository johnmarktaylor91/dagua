<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **drl**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

All 5 drl variants: **partial_match**. Median RMSD 0.121-0.213 vs igraph_drl.
Variants: drl_coarsen (0.127), drl_coarsest (0.132), drl_default (0.121),
drl_final (0.165), drl_refine (0.213). 92-96 samples each (good coverage).

## Source code locations

- Dagua: dagua/layout/ops/pipelines/drl.py, dagua/layout/ops/drl.py (if exists)
- Reference: /home/jtaylor/projects/_references/igraph/src/layout/drl/
  (drl_layout.c, drl_init.c, drl_layout_3d.c, drl_layout_threaded.c if present)

## Prior round work

- ROUND_19_diff_drl.md + PROMPT_20_fix_drl.md exist; R20 attempted node-acceptance
  rule alignment, improved 0.206->0.189 but missed commit threshold (reverted).
- R20 SUMMARY noted: 'drl edge-cutting alignment' as deferred work -- igraph
  removes selected long edge from only current node's neighbor map, dagua removes
  symmetrically. This is the canonical remaining lever.
- Also: classic_drl_final phase parameters identified as having FINAL preset
  mismatch but never targeted (R14 used default preset).

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/drl/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
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
- WRITE ONLY: the PLAN_*.md file in the specified round_31/drl/ dir
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
