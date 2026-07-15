<task>
r76-C4b-CONTINUATION: next bisection stage for connected SFDP residuals. Round 1 (READ FIRST,
in this worktree: .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_sfdp_conn_NOTES.md and r76_PROBE_sfdp_triage.md) found + FIXED the first divergence
(graphviz symmetrized-CSR neighbor order -> multilevel hierarchy parity; COMMITTED as 681370b
on this branch; RMSD collapsed 0.526->0.0025 hourglass, 0.384->0.0018 hexagonal, 0.090->0.019
planar_60). TWO representative graphs were UNTOUCHED by that fix and diverge at a LATER
stage: real_karate_34 (median RMSD 0.3898) and weighted_chain_20 (0.2390). Round-1 notes
point at prolongation output or spring-electrical iteration internals. Your job: bisect
those, fix if an op difference is named, floor-dossier ONLY if bisection exhausts.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sfdp-conn (branch r76/sfdp-conn, HEAD
09b8a28 -- round-1 fix committed). Work ONLY here. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

STEP 1 -- REBUILD THE INSTRUMENTED TRACE (round 1's /tmp/gv750-trace was cleaned up):
`mkdir -p /tmp/gv750-conn2 && git -C /home/jtaylor/projects/_references/graphviz archive
7.0.5 | tar -x -C /tmp/gv750-conn2` (NEVER dirty the reference clone; /tmp/gv750-disc may
exist and belongs to a PARALLEL task -- do not touch). Re-apply the round-1 instrumentation
pattern (described in the notes: GV_SFDP_TRACE=1 gated fprintf dumps -- symmetrized rows,
cluster maps, coarse sizes, coarsest random coords + K, first-3 iteration force norms + step
sizes, prolongation coords) and EXTEND it deeper for this round: per-iteration per-node force
components on the coarsest + first fine level, cooling/step updates, convergence exit
criteria, and prolongation interpolation inputs/outputs. DOT input via
dagua/eval/competitors/graphviz_competitor.py::_graph_to_dot; params
maxiter=500,theta=0.6,repulsiveforce=-1.0, seed 100 -- match round 1.

STEP 2 -- BISECT real_karate_34 AND weighted_chain_20: with hierarchies now matching (verify
first! if hierarchy/cluster maps DIFFER on these graphs, that is your divergence -- these two
were unchanged by the CSR fix, so their coarsening may hit a DIFFERENT ordering/tie-break
rule, e.g. weighted-edge handling in matching: weighted_chain_20 and real_karate_34 both have
edge WEIGHTS -- check how graphviz consumes weights in coarsening matching vs dagua), walk
the pipeline: coarsest init coords -> per-iteration forces/steps -> level prolongation ->
finer-level iterations. Name the FIRST diverging quantity (level, iteration, node, value).
Weighted-edge hypothesis is prime: hourglass/hexagonal/planar (fixed by CSR order) are
unweighted; karate + chain are weighted. Pinned source: `git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:lib/sfdpgen/<file>` (Multilevel.c
matching/weights, spring_electrical.c force loop, PriorityQueue.c if used by matching).

STEP 3 -- FIX OR FLOOR:
- Op difference found (expected): smallest gated fix scoped to graphviz-fidelity SFDP.
  Verify: RMSD on real_karate_34 + weighted_chain_20 (seeds 100-104) collapses comparably to
  round 1's wins; spot-check 3 more cluster graphs (weighted_karate_34, sparse_pair_50,
  real_lesmis_77); byte-identical unchanged-row gate (same row set as round 1); pytest -k
  sfdp green; ruff clean. Commit on pass.
- Bisection exhausts with every quantity matching to float rounding: run the 1-ULP
  perturbation experiment (nudge one coarsest init coord by 1 ULP, show final divergence
  pattern/magnitude reproduces observed gaps) and write the floor dossier; NO commit.

KNOWN PRE-EXISTING FAILURE (verified on develop, NOT yours, do NOT let it block commit):
tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest fails on
the base commit too. Exclude it (deselect or note it) -- a gate is green if everything EXCEPT
this known failure passes.

DELIVERABLES: append "## Round 2: post-hierarchy bisection" to r76_IMPL_sfdp_conn_NOTES.md
(trace tables, first-divergence naming, fix + 7.0.5 cites OR floor dossier, gate evidence,
commit sha). Conventional commits on r76/sfdp-conn; re-add/re-commit through ruff-format
until `git log` SHOWS them. No push/merge. NO AI attribution. ASCII only. Clean up
/tmp/gv750-conn2 at the end.
</task>
<completeness_contract>
Done = first post-hierarchy divergence NAMED with trace evidence AND (gated fix committed
with gates green, OR floor dossier with 1-ULP perturbation evidence and no commit). Never
claim floor without the perturbation experiment. Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Never write to /home/jtaylor/projects/_references/graphviz or
/tmp/gv750-disc. Never touch other engines, eval scoring, reference runners. Never modify
files outside this worktree except /tmp/gv750-conn2 scratch.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
