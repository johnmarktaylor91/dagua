<task>
r76-A2: make dagua's graphviz x-coordinate network-simplex assignment (stage A, landed r75 commit
e6ba3db) FAST enough for 100-500 node graphs. Today it renders binary_tree-class graphs instantly
but takes ~150s/seed on dense_pair_50-class inputs and >270s on ba_500 -- which left 9 combos
unmeasured in r75 (citation_dag_300, planar_60, protein_ppi_200, real_football_115,
real_lesmis_77, rgg_100, sbm_4x30, sbm_5x50, scale_free_ba_120 x classic_sugiyama_graphviz_fidelity)
and blocks the ba_500 tail. Read first:
- .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_sugiyama_xns_NOTES.md
- .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_mincross_NOTES.md (the
  "ba_500 Gate" section: RecursionError was fixed via iterative DFS in a since-lost patch --
  recover that idea; the patch text survives in r75_findings/r75_cx_mincross2.log, grep for
  _dfs_range_init / _dfs_cutval).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-xns-perf (branch r76/xns-perf). Work ONLY
here. Conventional commit (perf(sugiyama): ...). No push, no merge. Pre-commit ruff-format may
reformat: re-add and re-commit until `git log` SHOWS your commit. PYTHONPATH=$PWD for all runs.

CONSTRAINTS (hard):
- OUTPUT UNCHANGED where it already works: for binary_tree, bipartite_4_3_4, org_chart_1_5_4_8,
  center_port_backedge_hub (2 seeds each via benchmark path), positions must be BIT-IDENTICAL
  pre/post (torch.equal). The optimization must not alter pivot choices/tie-breaks -- pure
  algorithmic speedup of the same computation (profile-first: cProfile ONE dense_pair_50 seed;
  report where time goes BEFORE changing code).
- Reference semantics stay 7.0.5-faithful (git -C /home/jtaylor/projects/_references/graphviz
  show 7.0.5:lib/common/ns.c -- graphviz's own network simplex is fast at these sizes; port its
  algorithmic devices: cutvalue INCREMENTAL updates, tree-edge search order, enter/leave edge
  heuristics (search_size/S semantics), rather than inventing new ones).
LIKELY HOTSPOTS: naive cutvalue recomputation per pivot (graphviz updates incrementally along the
tree path); Python-level edge scans per iteration; the LR-balance pass. numpy-vectorize scans if
draw-order/tie semantics preserved.

TARGETS + VERIFICATION:
a. Profile evidence before/after: dense_pair_50 <=10s/seed; sbm_5x50 <=30s/seed; ba_500 1 seed
   <=240s (benchmark path, --timeout generous).
b. Bit-identity gate (above) + pytest tests/ -k "sugiyama or dot_rank" -x -q green.
c. Then run the 9 unmeasured combos x 5 seeds (seed-start 100) via scripts/run_benchmark.py into
   /tmp/r76_xns_perf_probe: all 45 ok, 0 timeouts; report per-graph seconds/seed.
Write .project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_xns_perf_NOTES.md
(same findings dir): profile, what changed, before/after timings, bit-identity evidence, commit.
</task>
<completeness_contract>
Done = profile-driven speedup committed; bit-identity 8/8; targets (a) met or the precise
remaining hotspot named with ns.c cites; 9-combo probe 45/45 ok; notes written; `git log` shows
the commit.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/xns-perf only. Do not change BK paths, mincross, eval code.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
