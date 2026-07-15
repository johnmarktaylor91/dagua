<task>
r77-B1: MAAR packing -- instrumented RUNNER trace (the step attempt-2 skipped). Context
(READ FIRST): .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_fmmm_disc_NOTES.md "Attempt 2". It named tie rules from SOURCE READING (MAARPacking
pairing-heap newest-push ties; OGDF qsort stable <=40) but porting them made random_dag_50
WORSE vs honest refs -- meaning the inferred decision sequence is wrong somewhere. 4 rows
remain (random_dag_50 fmmm steps10/100/200 + gem iters2000; quality already equal-or-better,
so this is pure fidelity-chasing per JMT's perfect-fidelity directive).

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-maar-trace (branch r77/maar-trace, off
develop). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

STEP 1 -- ACTUALLY RUN AN INSTRUMENTED RUNNER: copy scripts/ogdf_runner.cpp to
/tmp/maar-trace/, patch it with fprintf(stderr) dumps inside the packing call path (link
against the SAME OGDF build the committed runner uses -- figure out the build recipe from
scripts/ or the r75 rebuild commit 0817427; ogdf-src at /home/jtaylor/tools/ogdf-src).
Dump: per-CC bbox handed to the packer, the sort keys and FINAL sorted order,per-row
assignment decision, final offsets. Run fmmm steps10 + gem iters2000 on random_dag_50
(seeds 100-101). Dump dagua's packer decisions on identical inputs. Diff the DECISION
SEQUENCES -- the first differing decision is the real rule (source reading got it wrong
once already; trust only the trace).
STEP 2 -- PORT the traced rule (gated to the OGDF-family disconnected path; graphviz
packers untouched).

GATES (before commit): per-seed Procrustes RMSD vs honest refs (r76_refs, MAIN repo
read-only) drops decisively on random_dag_50 for fmmm AND gem (5 seeds, matched params);
zero regressions (byte-identical: 5 identical fmmm rows + 5 identical gem rows, 3 seeds; 2
connected probes each); pytest -k "fmmm or gem" green; ruff clean. KNOWN pre-existing
failures (must not block): test_bench_large; classic_fcose; double-border smoke. Commit on
r77/maar-trace; re-bench random_dag_50 fmmm+gem into
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_maar (0 errors).

DELIVERABLES: append "## Attempt 3: instrumented runner trace" to r76_IMPL_fmmm_disc_NOTES
.md (the decision-sequence diff, the real rule w/ cite, before/after RMSD, gate evidence,
commit sha OR the dossier showing the decision sequence depends on unobservable state
[e.g. allocator addresses] -- which would prove genuine non-portability). Clean /tmp
scratch. ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = the REAL decision sequence traced from a RUNNING instrumented runner AND (port
committed with gates green + bench, OR non-portability proven by the trace itself, e.g.
pointer-order-dependent ties shown in the dump). Source-reading-only conclusions are NOT
acceptable this round.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/maar-trace only. NEVER modify scripts/ogdf_runner (the
committed binary/source) or ogdf-src -- instrumented copies live in /tmp only. Never touch
graphviz packers, eval scoring. Bench write to benchmark_100seed_r77_maar only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
