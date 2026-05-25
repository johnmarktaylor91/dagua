<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **tsnet**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

All 5 tsnet variants: **partial_match**. Median RMSD 0.15-0.28 vs tsne_graph.
Variants: tsnet_default (0.267), perp5 (0.151), perp50 (0.270), steps200 (0.273),
steps2000 (0.276). 42-68 samples each (lower coverage than others).

## Source code locations

- Dagua: dagua/layout/ops/pipelines/tsnet.py, dagua/layout/ops/tsnet.py
- Reference: python -c 'import sklearn.manifold._t_sne; print(sklearn.manifold._t_sne.__file__)'
  -> sklearn t-SNE source. Plus tsne_graph reference adapter at
  dagua/eval/competitors/ (find via grep tsne_graph)

## Prior round work

- ROUND_18 already attempted; reverted because of dagua's torch RNG variance
  vs sklearn's near-zero seed-to-seed RMSD on tiny graphs.
- R20 SUMMARY: 4/5 small graphs already TOST equivalent at 0.5x of sklearn floor;
  remaining gap on parallel_multiedge_bundle.
- R30 commit 07b6d62 fixed grad_fn issue under no_grad context (autograd
  enable_grad). That fix wasn't in the 100-seed run, so post-R30 samples will
  improve. But algo divergences remain.

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/tsnet/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
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
- WRITE ONLY: the PLAN_*.md file in the specified round_31/tsnet/ dir
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
