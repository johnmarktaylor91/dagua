<task>
r77-M3: mds disconnected -- the LAST placement rule. M2 (this worktree, HEAD b029ab1; READ
its dossier section in .project-context/research/sprint_rng_matching/r75_findings/
r76_THIN_ROW_DOSSIERS.md) achieved EXACT RNG draw-count parity with igraph's DLA merge
(125,978==125,978 on random_dag_50; 316,682==316,682 on random_dag_200) by porting the
get_sphere() quadrant scan (incl the bounds typo) and equal-size component qsort ordering.
Yet same-process RMSD remains large -- with the draw stream aligned, exactly one (or few)
PLACEMENT rule(s) still differ. Find and port it.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mds-disc (branch r77/mds-disc). HARD RULE:
NO runtime igraph imports in dagua/layout (AST guard test exists -- keep it green);
installed igraph / instrumented /tmp builds are for OFFLINE tracing only.

METHOD: with draw counts equal, instrument BOTH sides (dagua DLA + instrumented igraph
/tmp venv build, merge_grid.c/DLA code) to dump PER-COMPONENT-PLACEMENT: component id/size,
walk start cell, walk path length, final collision cell, the PLACED coordinates (grid ->
final coords transform), and any post-placement updates (grid occupancy marks, centroid
shifts). On random_dag_50 seed 100 (SAME-PROCESS graph realization for any dagua-vs-
reference comparison -- hash bug caveat per r76_REFS_PROVENANCE.md), diff placement-by-
placement; the first differing PLACED COORDINATE names the rule (candidates: grid-cell ->
coordinate mapping (cell center vs corner, radius scaling), the sphere-scan pick WITHIN the
found cells, component anchor point (centroid vs bbox center vs first vertex), final
rescale/centering of the merged layout).
PORT the named rule natively.

GATES (before commit): placement-by-placement parity on the traced seed (all components,
coords equal to float64 tolerance); same-process RMSD vs igraph reference collapses
(<1e-6) on random_dag_50 + random_dag_200 + 2 more of the 6 target graphs (3 seeds each);
zero regressions (r75 bit-identity 9/9 probe set + 5 previously-identical mds rows
byte-identical; AST guard green); pytest -k mds green; ruff clean. KNOWN pre-existing
failures (must not block): test_bench_large; classic_fcose; double-border smoke. Commit on
r77/mds-disc; bench the 6 target graphs' classical_mds combos into
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_mds2 (seeds 100-199, 0
errors).

DELIVERABLES: append "## M3: placement rule" to r76_THIN_ROW_DOSSIERS.md (placement-diff
table, named rule w/ merge_grid.c cite, RMSD collapse evidence, gate evidence, commit sha,
bench Done line). ASCII. NO AI attribution. No push/merge. Clean /tmp scratch.
</task>
<completeness_contract>
Done = first differing placement NAMED AND (native port committed with RMSD collapse +
bench, OR a dossier showing the placement depends on non-reproducible state with the dump
shown). Draw parity is already achieved -- stopping short of coordinate parity requires
proof, not fatigue.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/mds-disc only. NEVER import igraph from dagua runtime.
Never modify installed igraph, other engines, eval scoring. Bench write to
benchmark_100seed_r77_mds2 only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
