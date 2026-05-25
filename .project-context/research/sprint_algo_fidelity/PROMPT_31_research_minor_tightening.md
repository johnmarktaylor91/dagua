<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **minor_tightening**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

TWO targets in one prompt (small):
- fa2_dissuade_hubs: partial_match, RMSD 0.104 (single variant; all other fa2
  variants are strong_equivalent).
- stress_sgd 4 variants: weak_equivalent. RMSD 0.04-0.05 (eps001, eps01,
  steps30, steps300). These are CLOSE to strong but at looser margin.

## Source code locations

- fa2: dagua/layout/ops/pipelines/fa2.py, dagua/layout/ops/fa2.py
  Reference: fa2-modified or fa2_ref pypi (find install path via
  python -c 'import fa2_ref; print(fa2_ref.__file__)' or grep)
- stress_sgd: dagua/layout/ops/pipelines/stress_sgd.py
  Reference: sgd2 (find via python -c 'import s_gd2; print(s_gd2.__file__)')

## Prior round work

- R21 + R22 + R23 had fa2 + stress_sgd work. classic_fa2 strong_equivalent for
  10 of 11 variants; only fa2_dissuade_hubs is the holdout.
- stress_sgd already at 0.04-0.05 RMSD; small algorithmic tightening could push
  to strong. Look at convergence/eps semantics, learning rate schedule.

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/minor_tightening/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
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
- WRITE ONLY: the PLAN_*.md file in the specified round_31/minor_tightening/ dir
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
