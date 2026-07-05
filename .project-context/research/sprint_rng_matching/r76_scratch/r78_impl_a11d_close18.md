<task>
r78-A11d: the LAST 18 igraph rows (close tier, d_R 0.01-0.1) -- graphs
dependency_graph_100, er_100, multi_component_80, parallel_cycles_4x5 (row list:
.project-context/research/sprint_rng_matching/r76_scratch/r78_igraph_close18.txt; two are
DISCONNECTED graphs -- component handling suspect). All prior igraph laws are merged on
develop (GLPK rank, BK alignment, ordinal conflict quirk, chain incidence order; dossiers
in r75_findings/r76_IMPL_igraph_NOTES.md -- READ the A3-A11c sections). These 18 survived
every law so far: bisect them.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-close18 -b r78/close18 develop`. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl.

METHOD (standard winning loop): per graph class (disconnected pair vs connected pair),
same-process probe vs installed igraph -> instrumented igraph /tmp venv dump (ranks,
orders, chains, BK runs, per-component handling + igraph's own component packing for the
disconnected two) -> first diverging quantity -> port gated to fidelity_mode="igraph" ->
iterate. Subprocess-batch reference calls (segfault workaround).

GATES (before commit): d_R < 0.01 on >=14 of the 18 (aim for all); zero regressions (the
257 bit-exact sample 10 rows x 3 seeds; graphviz rows 5-row sample; no-swiglpk fallback);
pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
pre-existing failures (must not block): the standard 6-item list. COMMITS ON r78/close18
AUTHORIZED AND REQUIRED on gate pass. Bench the 4 graphs x classic_sugiyama --variants
--max-nodes 0 --seeds 100 --seed-start 100 --workers 4 --timeout 3600 --watchdog-timeout
7200 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_close18
(0 errors).

DELIVERABLES: append "## A11d: the last 18" to r76_IMPL_igraph_NOTES.md (per-class
bisection, named laws w/ cites, before/after d_R, gate evidence, commit shas, bench line).
ASCII. NO AI attribution. No push/merge. Clean /tmp.
</task>
<completeness_contract>
Done = each of the 4 graph classes either ported to <0.01 (gates green) or carrying an
instrument-grade non-portable proof. This is the final igraph round of the campaign.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/close18 only. NEVER modify installed igraph; no runtime
igraph imports in dagua/layout. Bench write to benchmark_100seed_r78_close18 only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
