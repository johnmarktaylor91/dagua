<task>
r78-A12f: cluster x-stage, TERMINAL round -- the four-item checklist. Round 5 (READ
"## A12e" in .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_mincross_NOTES.md, this worktree) enumerated the COMPLETE structural difference
set between dagua's aux x-graph and dot's for clustered graphs:
  1. virtual/slack/boundary node INVENTORY (dot creates ln/rn cluster border nodes per
     cluster + contain_nodes aux edges with ND_lw+margin+GD_border widths -- see the
     contain_nodes excerpt in the dossier),
  2. cluster label compaction minlen reuse,
  3. keepout semantics,
  4. margin constraint counts.
Port ALL FOUR, in one pass, verifying the aux graph structurally after each (node count,
edge count, weight/minlen multiset vs the dot dump) before solving. No piecewise shipping:
the aux graph either matches structurally or the enumerated residual is the boundary.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-xstage (branch r78/xstage; round-5 dump
tooling patterns are described in the dossier -- rebuild the /tmp instrumented dot as
needed). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl. Pin: `git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:lib/dotgen/position.c` (contain_
nodes, keepout_othernodes, make_edge_pairs region, cluster borders) + cluster.c.

SHIP RULE: when the aux graph matches, the A9 output-space cluster pass is SUPERSEDED for
skeleton-mode rows (one mechanism ships); wrapper default-on for cluster-only DOT rows.

GATES (before commit): aux-graph structural parity (node/edge counts + weight/minlen
multisets) on the 3 dump graphs; solved x within tolerance vs dot dump; rendered d_R
improves on >=6/8 cluster probe rows, ZERO regressions (the standard byte-identity
samples); pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean.
KNOWN pre-existing failures (must not block): the standard 6-item list. COMMITS ON
r78/xstage AUTHORIZED AND REQUIRED on gate pass. Then the <=300 family bench to
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_xstage (0 errors).
IF a checklist item resists: STOP at the structural-parity stage and write which construct
cannot be mirrored and why (with the dump diff) -- that dossier is the sprint's terminal
disposition for the 20 cluster rows.

DELIVERABLES: append "## A12f: checklist port" to r76_IMPL_mincross_NOTES.md (per-item
port log w/ cites, structural parity tables, before/after d_R, gate evidence, commit shas,
bench line). ASCII. NO AI attribution. No push/merge. Clean /tmp.
</task>
<completeness_contract>
Done = four items ported + structural parity + gates green + committed + bench, OR the
terminal dossier naming the resisting construct with dump evidence. This is the LAST
cluster round of the sprint either way.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/xstage only. Never touch igraph paths, eval scoring,
runners. Bench write to benchmark_100seed_r78_xstage only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
