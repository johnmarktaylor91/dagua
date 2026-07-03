<task>
Implement the igraph-faithful disconnected-graph path for dagua's classical_mds pipeline:
per-component MDS + a literal port of igraph's DLA (diffusion-limited aggregation) component
merge. This was APPROVED-WITH-CHANGES by the sprint's adversarial critique (verdict 22 in
.project-context/research/sprint_rng_matching/r75_findings/r75_ADVERSARIAL_VERDICTS.md -- read it).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-mds-dla (branch r75/mds-dla @ 89ed3c3).
Work ONLY here. Conventional commits (fix(classical_mds): ...). No push, no merge.

CONTEXT (read these first):
- Port spec + RNG requirements: .project-context/research/sprint_rng_matching/r75_findings/
  r75_mds_tails_codex.md (finding 2 -- full merge_dla.c walkthrough with line cites) and
  r75_mds_tails_sonnet.md (which PROVED a literal Python port completes in 17.4s on the worst
  target, random_dag_200 with 202 components -- the old "it hangs" folklore is false).
- Reference source: /home/jtaylor/projects/_references/igraph/src/layout/mds.c (split at :223-280),
  merge_dla.c (:100-178 sphere/radius/sort/grid, :266-298 walk), merge_grid.c (:70-198 rasterized
  occupancy -- PRESERVE the quadrant loop quirks).
- Dagua entry: dagua/layout/ops/pipelines/classical_mds.py (_layout_igraph_classical_mds at :171,
  global distance fill via graph_utils.py:347).
- CRITICAL RNG note: the benchmark adapter wraps igraph in python random.Random(seed) via
  igraph.set_random_number_generator (dagua/eval/competitors/igraph_competitor.py:46). For
  benchmark parity your RNG_UNIF draws must come from random.Random(seed) in C-call order.

REQUIREMENTS (from the critique -- these are HARD GATES):
1. Gate strictly: new path ONLY when the graph is disconnected (len(components) > 1). The
   connected path must be BYTE-IDENTICAL before/after -- add a regression test that runs a
   connected graph through the pipeline pre/post and asserts exact tensor equality against a
   frozen expectation, plus run the existing classical_mds test suite.
2. Component discovery order = igraph's weak-component order (first unseen vertex). Row-reorder
   semantics must match igraph's vertex_order handling (mds.c:250-280).
3. Implement scalar/NumPy first (control-flow heavy); no torch vectorization cleverness. Use a
   small rng_unif(lo, hi) wrapper over random.Random. Add development-only step caps that RAISE
   (never fallback) if exceeded -- then set them generously high (e.g. 10M steps) as guardrails.
4. Target: distributional equivalence (rung-3), NOT bit-exactness, on the 16 disconnected target
   combos (list: .project-context/research/sprint_rng_matching/r75_findings/
   r75_targets_classical_mds.json -- rows with disconnected=true).
5. Do NOT reintroduce anything resembling TileToRows packing (r74 revert f342617 -- pure harm).

VERIFICATION LOOP (must pass before you finish):
a. pytest tests/ -k "classical_mds or mds" -x -q -- all green, plus your new tests.
b. Benchmark-path probe (MPLCONFIGDIR=/tmp/mpl): python3 scripts/run_benchmark.py --workers 2
   --timeout 120 --seeds 5 --seed-start 42 --variants --output-dir /tmp/r75_mds_probe
   --graphs multi_component_80,parallel_cycles_4x5,random_bipartite_60
   --engines classic_classical_mds_default,classic_classical_mds_igraph_fidelity
   Then compute normalized stress of dagua vs the saved igraph_mds reference positions (overlay
   dirs: eval_output/benchmark_100seed_seeded_refs has igraph_mds__for__* refs in the MAIN repo
   /home/jtaylor/projects/dagua/eval_output -- read-only) and show the stress gap SHRINKS vs the
   values in r75_targets_classical_mds.json for those graphs.
c. Connected regression: run 3 connected graphs (petersen_10, densenet_block, binary_tree) through
   the pipeline at 3 seeds pre-change (git stash) and post-change; assert byte-identical positions.
   Include the comparison script output in your final notes.
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_mds_dla_NOTES.md:
what changed, commits, probe numbers (before/after stress deltas per graph), open residuals.
</task>
<completeness_contract>
Done = DLA port implemented + gated, all tests green, connected path proven byte-identical,
probe shows material stress-gap reduction on >=2 of 3 probe graphs, notes file written. If the
walk semantics prove ambiguous on some grid quirk, match merge_grid.c literally and document.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/mds-dla only. Do not modify eval metrics/report code. Do not
touch the main repo checkout. Keep probe outputs under /tmp.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in the NOTES file. Stop only for
correctness-critical ambiguity.
</default_follow_through_policy>
