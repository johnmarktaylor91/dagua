<task>
r76-A1-FIXUP: the ONE sanctioned focused fixup for the graphviz mincross port (the original
r76 brief budgeted "this attempt + one focused fixup max" -- this is that fixup). The prior
attempt got ladder-a to 4/6 and NAMED the residual root cause. Your job: close the last two
graphs by fixing PRE-MINCROSS CONSTRUCTION PARITY, then run the full ladder and commit.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mincross2 (branch r76/mincross). It contains
UNCOMMITTED WIP from the prior attempt -- 3 modified files (dagua/layout/ops/_dot_mincross.py,
dagua/layout/ops/sugiyama.py, dagua/layout/ops/pipelines/sugiyama.py). DO NOT revert or
re-derive this WIP; it is your starting code. Work ONLY in this worktree. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl.

READ FIRST (all in the worktree):
1. .project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md --
   the prior attempt's full analysis. Current discriminator state: binary_tree 0=0,
   bipartite_4_3_4 36=36, hub_skip_superfan 2=2, dense_pair_50 271=271 (representative-chain
   merge fixed this exactly); FAILING: weighted_karate_34 dot=63 port=50 (pass-0 start 178 vs
   207), heavy_tail_weights_50 dot=50 port=59 (pass-0 start 96 vs 109).
2. .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_mincross_NOTES.md
   (attempts 1+2 inventory).

THE NAMED ROOT CAUSE (from the prior attempt -- both failing graphs diverge at PASS-0 START,
i.e. BEFORE median/transpose can matter; both have zero duplicate edges so chain-merge is not
implicated). Two concrete threads, in order:

(a) EDGE WEIGHTS ASYMMETRY: the benchmark DOT adapter
    (dagua/eval/competitors/graphviz_competitor.py:_graph_to_dot()) does NOT emit edge
    `weight=` attributes, so the reference `dot` runs with all weights=1. But dagua's graphviz
    rank assignment consumes graph.edge_weights. The prior attempt proved disabling those
    weights moves pass-0 starts closer (weighted_karate: 180 vs dot 178; heavy_tail: 91 vs
    dot 96) but is INSUFFICIENT alone. Implement the correct gating: under exact
    fidelity_mode="graphviz", rank assignment (and mincross ED_xpenalty/weight seeding) must
    mirror what the reference dot binary actually SEES (unit weights when the adapter omits
    weight=). Keep this scoped to graphviz fidelity -- do not change default/igraph paths.

(b) RANK / BUILD_RANKS SEEDING PARITY: after (a), bisect the remaining pass-0 delta in two
    steps. STEP 1 -- do the RANK ASSIGNMENTS match? Extract reference ranks by parsing real
    nodes' y-coordinates from `dot -Tdot` (or -Tplain) output (nodes cluster by rank; ranksep
    uniform); compare against dagua's post-rank assignment per node. If ranks differ, fix rank
    parity first (network simplex tie-breaks / balance / normalize -- cite
    lib/common/ns.c and lib/dotgen/rank.c as pinned below). STEP 2 -- if ranks match, the
    delta is the INITIAL ORDER seeding: graphviz build_ranks (mincross.c:1356-1414) seeds by
    iterating GD_nlist, the real-node list built by fast_node prepends/appends
    (fastgr.c:205-264), with virtual chain installation via class2.c:192-265. The prior
    attempt "recorded fast-node reverse creation order" -- verify the install order against
    the actual 7.0.5 semantics EXACTLY (prepend vs append, decompose/component order
    (dot -v prints per-pass counts; also compare pass-0 START, which is pure
    build_ranks+count). Iterate on the two failing graphs until pass-0 starts match dot
    (178 and 96), then final counts (63 and 50).

VERSION PIN: every graphviz claim via `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>` -- NEVER the working tree. DOT input built exactly as
dagua/eval/competitors/graphviz_competitor.py does; reference = installed dot 7.0.5 with -v.

LADDER (all must pass before commit; else document honestly and leave uncommitted):
a. Ordering-stage: match `dot -v` counts on >=5 of 6 calibration graphs (binary_tree,
   bipartite_4_3_4, hub_skip_superfan, weighted_karate_34, dense_pair_50,
   heavy_tail_weights_50).
b. Stage-A no-regression: binary_tree/bipartite_4_3_4/org_chart_1_5_4_8 stress vs reference at
   or below r75 stage-A values (r75_findings/r75_IMPL_sugiyama_xns_NOTES.md).
c. Rendered crossings move TOWARD reference on >=3/4 of dense_pair_50, weighted_karate_34,
   hub_skip_superfan, heavy_tail_weights_50 (benchmark path).
d. ba_500 1 seed completes <=300s AND rendered crossings improve >=2x from the 22344 baseline
   (the incremental-transpose perf work is already in your WIP).
e. igraph/default sugiyama byte-identical (5 seeds x default,tight on binary_tree +
   densenet_block); pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green.

DELIVERABLES: update r76_IMPL_mincross_NOTES.md in place (per-ladder numbers, what the fixup
changed, 7.0.5 cites, commit sha). Conventional commits on r76/mincross; re-add/re-commit
through ruff-format until `git log` SHOWS them. No push/merge. NO AI attribution in commits.
</task>
<completeness_contract>
Done = ladder a-e pass and committed, OR precise documented failure naming the exact unported
rule with 7.0.5 cites and NO commit. Never weaken a gate. A false pass is worse than an honest
failure -- gate_5 laundering checks will catch it downstream.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/mincross only. Never touch igraph/default ordering paths, BK
paths, eval code, scripts/ogdf_runner. Never modify files outside the worktree.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
