<task>
r78-A11b: BK round 2 -- make the dummy-chain rule PRINCIPLED and finish the last igraph
classes. Round 1 (READ: "## A11: BK second-order" in .project-context/research/
sprint_rng_matching/r75_findings/r76_IMPL_igraph_NOTES.md, this worktree; code UNCOMMITTED
in the worktree) proved dummy-chain id ordering is BK-visible and collapsed
densenet_block + width_skew_late_merge to d_R 0.0 (all variants) -- but shipped it behind
`N <= 20`, an arbitrary size gate added because the BROAD version regressed the sacred
karate row. A size gate is a probe-fit, not a rule. Fix it properly:

1. FIND THE REAL CONDITION: igraph creates dummy chains in a specific traversal order
   (read the igraph sugiyama source in /tmp/igraph-src, fetch matching version READ-ONLY:
   where are dummy vertices appended relative to edge iteration?). The correct port
   reproduces THAT order exactly for ALL graphs -- if the broad version regressed karate,
   the broad version's order was WRONG (karate is cyclic -> igraph's feedback-arc handling
   reorders edges before chain creation? -- that is the likely missing piece: chain order
   follows the POST-feedback-arc edge order, not the input order). Bisect karate's chain
   ids against an instrumented igraph build (/tmp venv, the standard pattern) and derive
   the size-free rule. Replace the N<=20 gate with it.
2. REMAINING CLASSES: hexagonal_lattice_42 + hub_skip_superfan -- with chains and ranks
   now exact, run the instrumented BK dump (per-direction candidate coords, median_4
   balancing) on one representative each; name and port the next quantity.

GATES (before commit): probe >=11/12 under d_R 0.01 with densenet/width_skew STAYING 0.0
and karate STAYING bit-exact; zero regressions (10-row bit-exact sample x 3 seeds
byte-identical; 5 graphviz rows byte-identical; no-swiglpk fallback green); pytest tests/
-k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN pre-existing failures
(must not block): the standard 6-item list. COMMITS ON r78/bk2 AUTHORIZED AND REQUIRED on
gate pass (generic no-commit guidance does NOT apply to this dispatched branch). Then
family bench --engines classic_sugiyama --variants --max-nodes 300 --seeds 100
--seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_bk2 (0 errors).

DELIVERABLES: append "## A11b: principled chain order" to r76_IMPL_igraph_NOTES.md (the
derived rule w/ source cite, karate bisection, before/after d_R table, gate evidence,
commit shas, bench line). ASCII. NO AI attribution. No push/merge. Clean /tmp scratch.
Segfault workaround: batch reference igraph calls in fresh subprocesses.
</task>
<completeness_contract>
Done = size gate REPLACED by the derived rule + gates green + committed + clean bench, OR
the bisection dossier proving the order depends on non-reproducible state. An N-threshold
in shipped code is not an acceptable endpoint.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/bk2 only. NEVER modify installed igraph; never touch
graphviz paths, eval scoring, runners. Bench write to benchmark_100seed_r78_bk2 only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
