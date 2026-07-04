<task>
r77-D1: DrL coarsening trace -- the last evidence-thin row. The r77 thin-row probe (READ
FIRST: .project-context/research/sprint_rng_matching/r75_findings/r76_THIN_ROW_DOSSIERS.md)
proved the flagged DrL row is NOT 1-ULP chaos: the divergence originates inside
DRLPhaseSolve (igraph's DrL phase schedule/coarsening), a portable op difference. Name it
and fix it if portable.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-drl-trace (branch r77/drl-trace, off
develop). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

STEP 1 -- INSTRUMENTED IGRAPH TRACE BUILD (the honest tool; sanctioned pattern -- 5th
trace build of this campaign): fetch the igraph C source matching the INSTALLED
python-igraph version to /tmp/igraph-drl-src; add fprintf(stderr) dumps in the DrL layout
code (src/layouts/drl/*: per-phase iteration counts, temperatures, attraction/damping
schedule values, coarsening decisions, first-3-iteration node updates); build python-igraph
against it IN A VENV in /tmp (pip install ./ from the patched source -- do NOT touch the
env's installed igraph). Run the flagged row's graph (from the dossier) at 1 seed through
BOTH the instrumented igraph and dagua's DrL pipeline with matched params; diff
phase-by-phase; name the first diverging quantity.
STEP 2 -- if the named difference is portable (schedule constant, order, RNG consumption),
port it gated to igraph-fidelity DrL; else write the non-portability dossier with the
trace shown.

GATES (before commit): flagged row's graph RMSD vs reference drops decisively (5 seeds);
zero regressions (5 previously-identical drl rows byte-identical, 3 seeds); pytest -k
"drl" green; ruff clean. KNOWN pre-existing failures (must not block): test_bench_large;
classic_fcose; double-border smoke. Commit on r77/drl-trace; re-bench the graph's drl
combos into /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_drl (seeds
100-199, 0 errors).

DELIVERABLES: append "## D1 DrL trace" to r76_THIN_ROW_DOSSIERS.md (phase-diff tables,
named quantity w/ source cite, fix or non-portability proof, gate evidence, commit sha).
ASCII. NO AI attribution. No push/merge. Clean /tmp scratch incl the venv.
</task>
<completeness_contract>
Done = first diverging phase quantity NAMED from the instrumented run AND (portable fix
committed + bench, OR non-portability dossier with trace evidence). Source-reading-only is
not acceptable.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/drl-trace only. NEVER modify the environment's installed
igraph (instrumented build lives in a /tmp venv). Never touch other engines, eval scoring,
reference runners. Bench write to benchmark_100seed_r77_drl only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
