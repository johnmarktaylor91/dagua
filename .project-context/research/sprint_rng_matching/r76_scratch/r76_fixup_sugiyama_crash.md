<task>
r76-A4d: fix the graphviz-fidelity sugiyama CRASH in the A1+A4b stack. The family bench
(seeds 100-199, benchmark path) crashed with `list index out of range` on EXACTLY 5 graphs,
all classic_sugiyama_graphviz_fidelity, all seeds: er_100, random_dag_50, regular_4_40,
rgg_100, sbm_5x50 (3 consecutive errors -> combo skipped; every other graph x variant ran
clean, 50500/51000 ok). Root-cause and fix the crash; do NOT touch behavior on graphs that
already work.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mincross2 (branch r76/mincross, HEAD 156cb25
= committed A1 ordering port + A4b box/units stack + dossiers). Work here; add commits on
top. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

REPRO (should crash immediately):
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python3 -c "
from dagua.eval.graphs import get_test_graphs
from dagua.eval.competitors.classic_competitor import ...  # find the classic_sugiyama
# graphviz_fidelity invocation path in dagua/eval/ and run graph er_100 seed 100"
-- or simpler, use scripts/run_benchmark.py --engines classic_sugiyama --variants
--graphs er_100 --seeds 1 --seed-start 100 --workers 1 --output-dir /tmp/crash_repro
and read the traceback (add temporary instrumentation to surface the full stack; the
benchmark harness reports only the message).

LIKELY SUSPECTS (from what the stack changed): the decompose(g,1)-style component DFS seed
ordering (A1) -- er_100/rgg_100/random_dag_50 can be DISCONNECTED (multiple components;
check), while regular_4_40/sbm_5x50 may have specific rank/degree structure; the
representative-chain merge; or the A4b box helper on graphs with specific label/degree
shapes. An index crash in a list traversal during build_ranks/decompose is the classic
empty-rank / missing-component case.

GATES (all before commit):
1. All 5 crash graphs run clean: run_benchmark --engines classic_sugiyama --variants
   --graphs er_100,random_dag_50,regular_4_40,rgg_100,sbm_5x50 --seeds 5 --seed-start 100
   -> 0 errors (output to a /tmp dir).
2. No behavior change on working graphs: byte-identical positions pre/post fix for
   binary_tree, dense_pair_50, weighted_karate_34, citation_dag_300 x graphviz_fidelity
   (3 seeds each) -- the fix must only affect the crashing paths (guard/correct indexing),
   not reorder anything that already works.
3. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
   PRE-EXISTING failures (must not block): test_bench_large hierarchy checkpoint;
   test_classic_competitor classic_fcose.
4. Commit on r76/mincross (conventional; NO AI attribution). Then BENCH THE 5 GRAPHS
   for real: run_benchmark --engines classic_sugiyama --variants --graphs
   er_100,random_dag_50,regular_4_40,rgg_100,sbm_5x50 --max-nodes 0 --seeds 100
   --seed-start 100 --workers 4 --timeout 3600 --output-dir
   /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r76_sugiyama_topup
   and confirm 0 errors in the final Done line.

DELIVERABLES: append "## A4d: crash fix" to
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md
(root cause w/ file:line, the fix, gate evidence, commit sha, bench Done line). ASCII only.
</task>
<completeness_contract>
Done = root cause named, fix committed, 5 graphs bench clean at 100 seeds with zero errors,
zero behavior change on working graphs. A crash is not disposable -- this gate cannot be
parked.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/mincross only. Never touch igraph/default paths, eval
scoring, reference runners. The topup bench write to eval_output/benchmark_100seed_
r76_sugiyama_topup is the ONLY main-repo write allowed.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
