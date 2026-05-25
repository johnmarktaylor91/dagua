<task>
Round 31 PARALLEL ADVERSARIAL RESEARCH for **umap**.

You are running in parallel with a rival agent from another lab on the SAME target.
We will compare your plan to theirs line-by-line. Be thorough, specific, accurate.
Read the actual source code, do not speculate.

## Current verdict (from 100-seed fidelity report)

All 6 umap variants: **partial_match**. Median RMSD 0.12-0.17 vs umap_graph.
Variants: umap_default (0.149), mindist001 (0.134), mindist05 (0.163), nn30 (0.174),
nn5 (0.120), spread2 (0.141). 77-84 samples each.

## Source code locations

- Dagua: dagua/layout/ops/pipelines/umap_layout.py (note: '_layout' suffix),
  dagua/layout/ops/umap.py
- Reference: python -c 'import umap; print(umap.__file__)' -> umap-learn package
  Key files: umap_.py (UMAP class, fit/transform), layouts.py
  (optimize_layout_euclidean), spectral.py (spectral init multi-component)

## Prior round work

- ROUND_21_DIFF_umap.md identified multiple items.
- R25 commit 7df7d6c fixed n_neighbors=N-1 cap; lifted from 3-of-5-graphs partial
  to all-5-graphs equivalent_at_1x on bounded subset.
- R22 + R23 prior commits: aac3ba3 (knn neighborhoods), 1760d31 (sampling),
  465a997 (weighted distances), 6d52627 (raw coords).
- Open per R21 diff: fuzzy simplicial set sigma/rho computation, multi-component
  spectral init, negative-sampling SGD details.

## Your mission

PURE RESEARCH. No code edits. No commits. No layout reruns.

Write `eval_output/algo_fidelity/round_31/umap/PLAN_$(whoami)_$(hostname)_$(date +%Y%m%d_%H%M%S).md`
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
- WRITE ONLY: the PLAN_*.md file in the specified round_31/umap/ dir
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
