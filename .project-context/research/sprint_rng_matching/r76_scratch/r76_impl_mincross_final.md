<task>
r76-A1: complete the graphviz mincross port for dagua sugiyama graphviz-fidelity. This is the
LAST major algorithmic gap in a two-month fidelity campaign (121 sugiyama rows divergent; the
graphviz family within them is ~50 and hinges on THIS). Two prior attempts got close and their
full analyses + patches are available -- you start from knowledge, not zero:
1. READ .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_mincross_NOTES.md
   (attempts 1+2 -- what was ported, what failed, source-cite inventory).
2. The attempt-2 CODE survives as patch text inside
   .project-context/research/sprint_rng_matching/r75_findings/r75_cx_mincross2.log (and
   r75_cx_mincross.log for attempt 1) -- grep for 'diff' / 'apply_patch' blocks and RECOVER the
   useful pieces (incremental transpose maps, iterative DFS -- note the dot_rank iterative-DFS
   part ALREADY landed via r76/xns-perf, so diff against current code first).
3. KNOWN RESIDUAL (from attempt 2): ordering-stage crossing counts already match `dot -v` on 3/4
   calibration graphs. The named blockers: (a) GD_nlist INSTALL ORDER -- graphviz iterates the
   real node list built by fast_node prepends (fastgr.c:205-264) and build_ranks seeds from it
   (mincross.c:1356-1414); attempt 2 approximated with reverse-creation-order; port it exactly.
   (b) REPRESENTATIVE-CHAIN MERGE -- class2.c:137-155 + fastgr.c:326-349 merge multi-edge virtual
   chains and ACCUMULATE ED_xpenalty; dagua keeps chains separate with unit penalties.
4. PERF MANDATE (new, from r76 xns work): the transpose loop is now the proven large-graph
   hotspot (faulthandler trace: _dot_mincross.py:166 _node_order_map inside _in_cross/_transpose).
   Port graphviz's incremental exchange deltas (in_cross/out_cross for the swapped pair only,
   mincross.c:632-688) -- required for the ba_500 gate.
VERSION PIN: every graphviz claim via `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>` -- NEVER the working tree.

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-mincross2 (branch r76/mincross). Work ONLY
here; PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl. Conventional commits; re-add/re-commit through
ruff-format until `git log` SHOWS them. No push/merge.

DISCRIMINATOR LOOP (attempt 2's proven method -- use it from the start):
Build DOT input exactly as dagua/eval/competitors/graphviz_competitor.py does, run
`dot -v 2>&1 | grep mincross` for the reference ordering-stage crossing count, compare your
port's post-ordering weighted count. Iterate per graph until they match or the exact unported
rule is named. Calibration set: binary_tree, bipartite_4_3_4, hub_skip_superfan,
weighted_karate_34 (the one attempt 2 missed: port 50 vs dot 63 -- note the port being BETTER
than dot is still a FAIL for fidelity), then dense_pair_50 (326 vs 271) and
heavy_tail_weights_50 (59 vs 50).

LADDER (all must pass before commit; else document and leave uncommitted):
a. Ordering-stage: match `dot -v` counts on >=5 of 6 calibration graphs.
b. Stage-A no-regression: binary_tree/bipartite_4_3_4/org_chart_1_5_4_8 stress vs reference at
   or below the r75 stage-A values (r75_IMPL_sugiyama_xns_NOTES.md).
c. Rendered crossings move TOWARD reference on >=3/4 of dense_pair_50, weighted_karate_34,
   hub_skip_superfan, heavy_tail_weights_50 (benchmark path).
d. ba_500 1 seed completes <=300s AND rendered crossings improve >=2x from the 22344 baseline.
e. igraph/default sugiyama byte-identical (5 seeds x default,tight on binary_tree+densenet_block);
   pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green.
Write .project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md:
per-ladder numbers, what was ported (7.0.5 cites), recovered-vs-new code, commit sha.
This is r76's attempt (budget: this attempt + one focused fixup max). An honest documented
failure is acceptable; a false pass is not.
</task>
<completeness_contract>
Done = ladder a-e pass and committed, OR precise documented failure naming the unported rule
with 7.0.5 cites and NO commit. Never weaken a gate to pass it.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/mincross only. Never touch igraph/default ordering, BK paths,
eval code, scripts/ogdf_runner.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
