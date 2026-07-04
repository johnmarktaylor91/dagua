<task>
r76-A3: igraph sugiyama tie-break parity -- the LAST major open work item of a two-month
fidelity campaign. The igraph-mode sugiyama family has 30 bit-exact + 6 near rows already,
but 227 rows sit far from the reference (mode-B d_R >= 0.1) and 74 close (<0.1). The r75
research (verdicts 19-21) named three candidate tie-break classes; your job: bisect, port,
flip as many rows as possible.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mincross2 (branch r76/mincross -- JUST
MERGED to develop; continue committing on this branch, it will be re-merged). The graphviz
sugiyama stack lives here too -- DO NOT touch graphviz-fidelity paths; igraph-mode code
paths only. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

READ FIRST:
- .project-context/research/sprint_rng_matching/r75_findings/r75_sugiyama_codex.md AND
  r75_sugiyama_sonnet.md (dual-lab research on the igraph family)
- r75_findings/r75_ADVERSARIAL_VERDICTS.md verdicts 19-21 (the named tie classes:
  (a) LP-SOLVER TIE PARITY: igraph's Sugiyama rank assignment solves an LP whose objective
  is ALL-ZERO on feedback-free DAGs -> the SOLVER'S tie-break determines ranks; installed
  igraph 1.0.0 uses GLPK; dagua's LP path differs (HiGHS-like behavior). (b) BK
  ordinal-edge conflict quirk. (c) qsort ties in ordering.)
- r75_findings/r75_IMPL_lp_objective_NOTES.md (the r75 LP-objective work that landed)

REFERENCE PIN: the INSTALLED igraph 1.0.0 runtime (python-igraph in this env) is the
reference -- version-pin claims via runtime traces (instrument igraph via its C source only
for reading; the adapter is dagua/eval/competitors/igraph_competitor.py). The reference is
DETERMINISTIC for sugiyama -- mode-B comparisons; d_R is the fidelity number.

METHOD -- BISECT FIRST (the method that closed 4 engines this sprint; do NOT guess):
1. Pick 3 representative far rows (e.g. from distinct graph classes: a lattice
   [hexagonal_lattice_42 d_R~?], a social graph [real_karate_34 igraph variants], a
   layered DAG [width_skew_late_merge]) + 1 close row. For each: compare stage-by-stage vs
   the installed igraph run -- rank assignment per node FIRST (if ranks differ, the LP
   tie-break is the cause -- verify by checking whether the objective is degenerate/all-zero
   on that graph), then ordering sweeps (median/transpose equivalents + qsort tie behavior),
   then x-coordinates (BK conflict handling).
2. Name the first divergent stage per row class; port the named rule (e.g. mirror GLPK's
   simplex pivoting tie-break on degenerate objectives -- likely implementable as a
   deterministic tie-break rule on the rank LP solution, NOT by vendoring GLPK; if the
   only faithful route is vendoring a solver, STOP and write the blocker analysis instead).
3. Iterate: fix -> re-run the 4 probe rows -> expand to a 10-row probe set -> full check.

GATES (all before commit; else honest dossier):
1. d_R improves materially on >=7 of a 10-row probe set spanning the far tier (report
   before/after d_R table); at least 3 rows reach d_R < 0.01.
2. Zero regressions: the 30 existing bit-exact igraph rows stay bit-exact (byte-identical
   positions, 3 seeds each on a 10-row sample); graphviz-fidelity rows byte-identical
   pre/post (5-row sample -- you must not disturb the merged stack).
3. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; ruff clean. KNOWN
   PRE-EXISTING failures (must not block): test_bench_large hierarchy checkpoint;
   test_classic_competitor classic_fcose.
4. ON PASS: conventional commits on r76/mincross; then bench the full igraph family:
   run_benchmark --engines classic_sugiyama --variants --max-nodes 300 --seeds 100
   --seed-start 100 --workers 5 --timeout 3600 --output-dir
   /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r76_igraph_fix
   and confirm the Done line has 0 errors (the graphviz_fidelity crash was fixed in
   aeaf194 -- if you see crashes, that is YOUR regression, fix it).

DELIVERABLES: write .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_igraph_NOTES.md (bisection tables per row class, named rules w/ igraph source
cites, before/after d_R table, gate evidence, commit shas, bench Done line). ASCII only.
NO AI attribution in commits. No push/merge.
</task>
<completeness_contract>
Done = gates 1-4 pass with commits and the family bench clean, OR a dossier naming exactly
which tie-break rule is non-portable (e.g. requires vendoring GLPK) with bisection evidence
per row class, and NO commit. An honest partial (some row classes fixed + committed, others
dossiered) is acceptable ONLY if gate 2 holds and each unfixed class has its named rule.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/mincross only. NEVER touch graphviz-fidelity code paths,
eval scoring, reference runners, other engines. The family bench write to
eval_output/benchmark_100seed_r76_igraph_fix is the only main-repo write allowed.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
