<task>
r77-M5: mds-disc FINISHER -- gate, bench, and commit the M2-M4 chain. M4 (READ its section
in .project-context/research/sprint_rng_matching/r75_findings/r76_THIN_ROW_DOSSIERS.md, in
this worktree) proved via C-side instrumented dump that DLA placement fully matches, found
+ported the eigensolve uplo='U' rule, and achieved same-process parity on 5/6 probes. The
single residual (random_dag_200 seed 102) is igraph_layout_align()'s eigenvector SIGN
selection from DSYEVR -- the LAPACK eigensign class ALREADY dispositioned for connected mds
("proven member of reference equivalence class", r76_FLOOR_DOSSIERS.md; vendoring excluded
by JMT ruling). Your job: finish the line -- verify, gate, bench, commit, document.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mds-disc (branch r77/mds-disc, HEAD
b029ab1 + uncommitted M3/M4 changes in classical_mds.py + tests). HARD RULE: no runtime
igraph imports in dagua/layout.

WORK:
1. Review the uncommitted diff: KEEP the M3 alignment port + M4 uplo='U' + scalar
   early-exit scan; REMOVE any half-finished "attempted align eigensign branch" if it does
   not provably match DSYEVR behavior (do not ship speculative sign heuristics -- the
   eigensign residual is dispositioned as equivalence-class, not patched).
2. EIGENSIGN RESIDUAL EVIDENCE (cheap, completes the dossier): for random_dag_200 seed 102,
   show that applying the appropriate sign flip/reflection within the align eigenbasis maps
   dagua's layout onto igraph's to tight tolerance (the equivalence-class proof pattern
   from r76_FLOOR_DOSSIERS.md). Also note whether the benchmark's registration treats
   reflections as equivalent (check the scorer's Procrustes: rotation-only or full
   orthogonal?) -- if reflections are modded out, this residual is invisible to scoring;
   say so explicitly.
3. GATES: same-process parity probe green on 5/6 (document #6 as eigensign); zero
   regressions (r75 bit-identity 9/9 set + 5 previously-identical mds rows byte-identical;
   AST no-igraph guard; pytest -k mds green); ruff clean; mypy dagua/cli.py. KNOWN
   pre-existing failures (must not block): test_bench_large; classic_fcose; double-border
   smoke; test_50_node_dag; graphopt seed-matrix.
4. COMMIT clean conventional commits on r77/mds-disc (alignment port; uplo fix; scan
   early-exit; tests; dossier). Then bench the 6 M1-target graphs' classical_mds combos
   into /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_mds2 (seeds 100-199,
   --max-nodes 0, 0 errors).
5. CLEANUP: extract a <=100-line representative excerpt of the M4 trace diff into the
   dossier, then DELETE /tmp/dagua_m4_igraph/traces (2.3G; disk is at 89%).

DELIVERABLES: append "## M5: close-out" to r76_THIN_ROW_DOSSIERS.md (final parity table,
eigensign equivalence-class evidence + reflection-handling note, gate evidence, commit
shas, bench Done line). ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = M2-M4 chain committed with gates green + clean bench + eigensign evidence + trace
cleanup. This CLOSES the mds-disc line.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/mds-disc only. Never import igraph from dagua runtime;
never modify installed igraph. Bench write to benchmark_100seed_r77_mds2 only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
