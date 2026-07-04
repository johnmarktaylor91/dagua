<task>
r77-A10: WIRE A8/A9 INTO THE BENCHMARK PATH -- pure plumbing, the last graphviz item. The
A8 edge-label port and A9 cluster port are merged on develop and verified by direct
pipeline probes, but the fresh family bench (benchmark_100seed_r77_sugiyama_final) produced
positions BIT-IDENTICAL to the pre-A8/A9 bench on every label/cluster graph
(edge_label_braid d_R 0.6016 unchanged; all cluster rows unchanged). Conclusion: the
benchmark harness path (scripts/run_benchmark.py -> dagua/eval/competitors/
classic_competitor.py spec for classic_sugiyama_graphviz_fidelity) never passes the new
edge-label-size / cluster metadata that A8/A9 gated on. Find the wiring gap and close it.

REPO: /home/jtaylor/projects/dagua (develop, HEAD ~fd8c9d0+). Work on a NEW branch
r77/a10-wiring in a worktree: `git worktree add ~/.claude/worktrees/dagua-a10 -b
r77/a10-wiring develop`. PYTHONPATH=$PWD (worktree); MPLCONFIGDIR=/tmp/mpl.

METHOD:
1. Trace the benchmark invocation chain for classic_sugiyama_graphviz_fidelity: which
   callable does _CLASSIC_LAYOUT_SPECS map to, what kwargs reach layout_sugiyama_pipeline,
   and where A8's `edge_label_sizes`/A9's `clusters` + `graphviz_apply_cluster_constraints`
   opt-ins are (or are not) computed. The A8/A9 wrapper helpers exist
   (_graphviz_dot_edge_label_sizes, the cluster-only DOT classifier) -- find why the spec
   path does not call them.
2. Wire them: the benchmark engine invocation must compute and pass the metadata EXACTLY
   per the wrapper's classification rules (label-only DOT -> edge_label_sizes; cluster-only
   DOT -> clusters + opt-in flag; mixed -> NEITHER, per A9b's guard). No behavior change
   for any other engine or fidelity mode.
3. VERIFY (the decisive check the codex probes already passed): bench edge_label_braid,
   nested_cluster_label_stack, interleaved_cluster_crosstalk, kitchen_sink_platform_graph,
   clustered_longlabel_handoffs, small_label_storm x classic_sugiyama_graphviz_fidelity
   (5 seeds) into /tmp scratch; positions for label/cluster graphs must now DIFFER from the
   r77_sugiyama_final dir's, and quick d_R vs graphviz_dot reference (offline adapter) must
   land near the A8/A9 probe values (edge_label_braid and the 5 cluster rows improving,
   small_label_storm byte-identical per A9b).

GATES (before commit): the step-3 table (old d_R -> new d_R per graph; small_label_storm
unchanged); byte-identity on 5 plain graphviz rows + 3 igraph rows (wiring must not touch
them); pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
pre-existing failures (must not block): test_bench_large; classic_fcose; double-border
smoke; test_50_node_dag; graphopt seed-matrix; test_classify_early_exit. Commit on
r77/a10-wiring. Then FULL family bench: run_benchmark --engines classic_sugiyama --variants
--max-nodes 300 --seeds 100 --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout
7200 --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_sugiyama_
wired -- 0 errors on the Done line.

DELIVERABLES: append "## A10: benchmark wiring" to
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md (the
gap, the fix, the step-3 table, gate evidence, commit sha, bench Done line). ASCII. NO AI
attribution. No push/merge.
</task>
<completeness_contract>
Done = the wiring gap NAMED + fixed + step-3 verification table showing the A8/A9 gains
reaching the benchmark path + clean full bench. This is plumbing; an honest failure here
means the A8/A9 code has a deeper activation problem -- name it precisely if so.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/a10-wiring only. Never change engine algorithm code --
wiring/metadata plumbing only. Bench writes to /tmp scratch + benchmark_100seed_r77_
sugiyama_wired only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
