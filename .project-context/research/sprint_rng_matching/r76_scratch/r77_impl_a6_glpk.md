<task>
r77-A6: igraph sugiyama GLPK PARITY VIA OPTIONAL DEPENDENCY -- JMT-AUTHORIZED. The r76 A3
work (READ FIRST: .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_igraph_NOTES.md) proved ~222 igraph-family rows diverge FIRST at rank/coordinate
LP solutions whose objective is degenerate, so the SOLVER's tie-break decides the answer;
SciPy HiGHS/simplex do NOT match installed igraph's bundled GLPK. JMT has now authorized
the middle path previously excluded as "vendoring": add GLPK AS AN OPTIONAL DEPENDENCY
(swiglpk) and solve the SAME LP the SAME way igraph does -- precisely analogous to the
umap-numba decision (match the reference's library rather than reimplement it). PyTorch
stays the only REQUIRED dependency: no swiglpk -> current behavior, unchanged.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-igraph-glpk (branch r77/igraph-glpk, off
develop). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

STEP 0 -- ENV: pip install swiglpk into the current conda env (PyPI, sanctioned). Verify
GLPK version: `python3 -c "import swiglpk; print(swiglpk.glp_version())"` vs the GLPK
version bundled in installed python-igraph 1.0.0 (check igraph's docs/source -- fetch the
igraph C source tarball matching the installed version to /tmp/igraph-src for READING ONLY;
python3 -c "import igraph; print(igraph.__version__)" and the C core version via
igraph.version). If versions differ materially, note it and proceed -- GLPK's simplex is
stable across minor versions; verify empirically at step 2.

STEP 1 -- TRACE THE REFERENCE LP: read igraph's sugiyama implementation in the C source
(src/layouts/sugiyama.c or similar; find where it builds the GLP problem: objective,
constraint rows, column bounds, glp_simplex parameters/order). Document the exact problem
construction and solver invocation.

STEP 2 -- PORT: in dagua's igraph-fidelity sugiyama path, when swiglpk is importable,
construct the IDENTICAL GLP problem (same row/col order, same coefficients, same solver
params) and consume its solution exactly as igraph does. Optional-import pattern mirroring
the umap-numba precedent (dagua/layout/ops/umap.py:18-30). Fallback (no swiglpk) = existing
code path byte-for-byte.

GATES (before commit):
1. Draw-level parity: on 4 probe graphs (real_karate_34, moe_router_sparse [the named
   LP-divergent rows], hexagonal_lattice_42, width_skew_late_merge), the LP solution vector
   (ranks/coords) matches installed igraph's EXACTLY; then full-layout d_R vs installed
   igraph collapses (target <0.01) on >=6 of a 10-row probe from the igraph far tier
   (per_combo_r76.jsonl, MAIN repo read-only).
2. Zero regressions: the 60 bit-exact igraph rows stay bit-exact (byte-identical positions,
   10-row sample x 3 seeds); graphviz-fidelity rows byte-identical (5-row sample);
   WITHOUT swiglpk (uninstall or monkeypatch the import), all sugiyama tests still pass and
   positions match current develop byte-for-byte (3 probe rows).
3. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green (add tests: with-glpk
   parity on a small graph + without-glpk fallback); ruff clean. KNOWN pre-existing
   failures (must not block): test_bench_large; test_classic_competitor classic_fcose;
   test_cosmetic_node_features double-border.
4. Commit on r77/igraph-glpk. Then FULL igraph-family bench:
   run_benchmark --engines classic_sugiyama --variants --max-nodes 0 --seeds 100
   --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir
   /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_igraph_glpk -- 0 errors.
   Also add swiglpk to pyproject optional extras (e.g. [igraph-fidelity] or the existing
   fidelity extra -- follow the repo's extras convention).

DELIVERABLES: .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_igraph_NOTES.md -- append "## A6: GLPK parity via optional dependency" (LP
construction cites, version check, parity tables, d_R before/after, gate evidence, commit
shas, bench Done line). ASCII only. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = draw-level LP parity proven AND gates 1-4 with commits + clean bench, OR a dossier
naming exactly why swiglpk's GLPK cannot reproduce installed igraph's solution (version
basis difference, presolve divergence -- with the two solution vectors shown) and NO
commit. Partial (some LP call sites ported, others dossiered) acceptable if gate 2 holds.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/igraph-glpk only. pip install ONLY swiglpk from PyPI.
Never touch graphviz-fidelity paths, eval scoring, reference runners. Never modify
installed igraph. Bench write to benchmark_100seed_r77_igraph_glpk is the only main-repo
write allowed.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
