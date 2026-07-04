<task>
r77-M4: mds disconnected -- FINAL round, ONE job only: C-SIDE INSTRUMENTED PLACEMENT DUMP.
Rounds M1-M3 (worktree HEAD b029ab1 + uncommitted M3 alignment port -- KEEP it) achieved:
component BFS order ported, quadrant-scan collision + qsort ported (RNG draw counts EXACT:
125978==125978), igraph_layout_align() binding post-step ported. Same-process parity probe
STILL fails (rmsd 554 on random_dag_50 seed 100): the raw per-component PLACEMENT diverges.
M3 inferred from binding source instead of instrumenting C placement -- that gap is your
entire mandate.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mds-disc (branch r77/mds-disc). HARD RULE:
no runtime igraph imports in dagua/layout (AST guard stays green).

DO EXACTLY THIS:
1. Build an instrumented python-igraph in a /tmp venv (fetch C source matching installed
   version; add fprintf(stderr) in merge_dla.c/merge_grid.c placement path: per component
   -- id, size, sort position, walk start angle/cell, every walk step's cell, termination,
   the FINAL placed (x,y) BEFORE and AFTER any scaling; plus the singleton fast-path if one
   exists). pip install ./ into the venv only.
2. Run random_dag_50 seed 100 layout("mds") under it; capture the dump.
3. Dump the SAME quantities from dagua's DLA port (same process/graph realization).
4. DIFF placement-by-placement. The first differing quantity IS the rule. Candidates the
   prior rounds could not distinguish: singleton placement fast-path; walk start-angle
   distribution; cell->coordinate transform (center vs corner, radius factor); component
   sort position consumed differently by placement than by walk order; post-placement grid
   occupancy radius.
5. PORT the named rule natively; re-run the parity probe: same-process rmsd on
   random_dag_50 + random_dag_200 (seed 100, 3 seeds if first passes) must COLLAPSE
   (<1e-6 target; document honestly if it lands near-but-not-at, e.g. float32 residue).

GATES (before commit): parity probe collapse as above; zero regressions (r75 bit-identity
9/9 set + 5 previously-identical mds rows byte-identical; AST guard; pytest -k mds green);
ruff clean. KNOWN pre-existing failures (must not block): test_bench_large; classic_fcose;
double-border smoke; test_50_node_dag; graphopt seed-matrix. On pass: commit EVERYTHING
(M3 alignment port + M4 rule) as clean commits on r77/mds-disc; bench the 6 target graphs
into /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_mds2 (seeds 100-199,
0 errors). On fail: append the placement-diff dossier (the dump IS the deliverable) and
leave uncommitted -- this closes the mds-disc line either way.

DELIVERABLES: append "## M4: C-side placement trace" to r76_THIN_ROW_DOSSIERS.md
(placement-diff table, named rule w/ C cite, parity numbers, gate evidence, commit shas OR
park rationale). ASCII. NO AI attribution. No push/merge. Clean /tmp scratch + venv.
</task>
<completeness_contract>
Done = the C-side placement dump EXISTS and is diffed (non-negotiable) AND (port committed
with parity collapse + bench, OR the dossier shows the diverging quantity is
non-reproducible with the dump attached). This is the final mds-disc round.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/mds-disc only. NEVER import igraph from dagua runtime;
never modify installed igraph. Bench write to benchmark_100seed_r77_mds2 only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
