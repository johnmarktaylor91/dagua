<task>
Implement the graphviz mincross completion (phase 1) for dagua's sugiyama graphviz-fidelity
variants: omega-weighted crossing counts + build_ranks-style initial ordering + the pass 0/1/2
schedule. APPROVED by the adversarial critique (verdicts 15 and 16 in
.project-context/research/sprint_rng_matching/r75_findings/r75_ADVERSARIAL_VERDICTS.md -- read it,
plus r75_sugiyama_codex.md finding F3 for the full delta inventory). The x-coordinate
network-simplex port (stage A) landed earlier today (commit e6ba3db) -- build on it.

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-mincross (branch r75/mincross @ develop).
Work ONLY here. Conventional commit (feat(sugiyama): ...). No push, no merge. Pre-commit runs
ruff-format; if it reformats, re-add and re-commit until the commit lands.
NOTE: run benchmarks/tests with PYTHONPATH=$PWD or they import the installed main checkout.

REFERENCE (version-pinned: `git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`
-- NEVER read the working tree, it is a newer version):
- 7.0.5:lib/dotgen/mincross.c -- pass structure (~:690-748 area), init via build_ranks per pass
  (~:815-855, :1212-1286), constants MinQuit/MaxIter/Convergence (~:157-160, defaults ~:1944-1952),
  weighted crossing counts via virtual_weight()/ED_xpenalty (~:1858-1888), transpose/exchange.
- 7.0.5:lib/dotgen/class2.c -- virtual-chain creation calls virtual_weight (~:84-95).
Verify every line range yourself against 7.0.5 -- the codex research report may have HEAD drift.

DAGUA SIDE:
- dagua/layout/ops/_dot_mincross.py -- the partial mincross port (single pass schedule, unit
  crossing counts, caller-supplied initial order).
- dagua/layout/ops/sugiyama.py -- initial order is currently a plain sorted layer
  (search for the ordered_layers construction feeding graphviz_mincross), and the graphviz
  pipeline wiring in dagua/layout/ops/pipelines/sugiyama.py.

SCOPE (this phase; do NOT attempt left2right flat/cluster constraints or full class2 multi-edge
merge -- that is the next phase):
1. OMEGA WEIGHTS: per-edge crossing penalty per the 7.0.5 table (endpoint-type pairs:
   real-real=1, virtual-real=2, real-virtual=2, virtual-virtual=4... verify exact table and
   orientation in virtual_weight()) x the edge's weight. Crossing objective = sum of
   xpenalty-weighted crossings, used consistently in mincross AND transpose accept/reject.
2. INIT ORDER: graphviz build_ranks-style initial ordering (BFS from source nodes in graph input
   order, two seedings for pass 0 and pass 1 -- verify exact behavior incl. the pass-dependent
   traversal) instead of sorted-by-node-id.
3. PASS SCHEDULE: passes 0/1 (limited iterations, different init) then pass 2 with MaxIter,
   MinQuit, Convergence=.995 semantics per 7.0.5.
4. Gate ALL of it to the graphviz fidelity mode; igraph/default ordering paths byte-identical.

VERIFICATION LADDER (benchmark path, PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl, references read-only
from main repo eval_output overlay dirs -- graphviz_dot__for__classic_sugiyama_graphviz_fidelity
positions in benchmark_100seed_escalation_final or seeded_refs):
a. NO-REGRESSION on stage-A wins: binary_tree, bipartite_4_3_4, org_chart_1_5_4_8 -- stress gap
   vs reference must stay at or below the stage-A values recorded in
   r75_findings/r75_IMPL_sugiyama_xns_NOTES.md.
b. CROSSING TARGETS (from r75_targets_sugiyama.json, graphviz-fidelity rows failing crossings):
   dense_pair_50 (D=391 vs R=331), weighted_karate_34, hub_skip_superfan, heavy_tail_weights_50 --
   1 seed each: dagua crossing count must MOVE TOWARD the reference count; report before/after.
c. SCALE SPOT-CHECK: ba_500, 1 seed, classic_sugiyama_graphviz_fidelity (timeout 300s): report
   crossing count before/after (baseline dagua ~22344 vs reference ~2805 -- expect a large drop;
   any improvement >2x is a pass for this phase, full closure may need the next phase).
d. REGRESSION GATE: classic_sugiyama_default + classic_sugiyama_tight (igraph family) 5 seeds on
   binary_tree + densenet_block: positions byte-identical pre/post (torch.equal). pytest tests/
   -k sugiyama -x -q green; extend tests for the omega table + init order (unit-level, no
   graphviz binary dependency).
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_mincross_NOTES.md:
ladder numbers before/after, what was ported vs deferred (with 7.0.5 line cites), commit sha.
</task>
<completeness_contract>
Done = omega weights + init order + pass schedule landed gated, ladder (a) no regression,
(b) all 4 move toward reference, (c) ba_500 improves >2x or you name the exact blocking rule with
7.0.5 cites, (d) byte-identical igraph paths + tests green, notes written, committed.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/mincross only. Never touch igraph/default ordering behavior,
eval metrics, or other engines. Main repo is read-only reference data.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
