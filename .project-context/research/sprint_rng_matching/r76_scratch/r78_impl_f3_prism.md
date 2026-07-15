<task>
r78-F3: FDP prism overlap-expansion parity -- the named stage behind 14 rows (5
DIVERGENT_NAMED_CAUSE + 9 SUPERIOR_DISTINCT, all classic_fmmm_graphviz_fdp_fidelity). R2
(READ: the R2 section of .project-context/research/sprint_rng_matching/r75_findings/
r78_RESIDUAL_MOP.md + artifacts in ../r78_evidence/) proved fdp_tLayout MATCHES on
representatives; the first divergence is fdp_xLayout's overlap expansion
(overlap="9:prism", tries=9): graphviz runs the PRISM proximity-stress overlap remover
(Gansner+Hu; lib/neatogen/overlap.c + GTS-based Delaunay; pin via `git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`) before packing.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-prism -b r78/prism develop`. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl.

SCOPE JUDGMENT FIRST (30 min): read what prism actually does at 7.0.5 (proximity graph
via Delaunay triangulation, iterative stress-model scaling until overlaps resolve, tries
capped at 9) and what dagua's fdp-fidelity path currently does after tLayout. Determine:
(a) does dagua already have overlap-removal ops that can be made prism-faithful? (b) is
Delaunay available via torch/scipy without GTS (scipy.spatial.Delaunay is in-env -- using
SCIPY for the triangulation is acceptable: it is a computational primitive, not the
reference package)? If the port is tractable, DO IT, gated to the graphviz_fdp fidelity
path. If the GTS-specific behavior (its particular triangulation/edge handling) proves
load-bearing beyond scipy-equivalence, STOP and write the boundary dossier with the
instrumented comparison (prism input/output dumps from the R2-style /tmp build vs dagua).

GATES (before commit): d_R/RMSD improves decisively on >=3 of the 5 divergent fdp rows AND
>=5 of the 9 superior rows move toward reference (benchmark path, 5 seeds); zero
regressions (5 previously-identical fmmm rows byte-identical; connected fmmm probes
unchanged); pytest -k "fmmm" green; ruff clean. KNOWN pre-existing failures (must not
block): the standard 6-item list. COMMITS ON r78/prism AUTHORIZED AND REQUIRED on gate
pass. Bench the 14 rows' graphs into
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_prism (seeds 100-199, 0
errors).

DELIVERABLES: append "## F3: prism parity" to r78_RESIDUAL_MOP.md (scope judgment, port w/
cites OR boundary dossier w/ dumps, before/after, gate evidence, commit shas). ASCII. NO
AI attribution. No push/merge. Clean /tmp scratch.
</task>
<completeness_contract>
Done = prism-faithful port committed w/ gates green + bench, OR the boundary dossier with
instrumented prism input/output comparison proving GTS-specific behavior is load-bearing.
The scope judgment must be explicit either way.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/prism only. scipy is permitted as a computational
primitive; GTS/graphviz must NEVER be invoked from dagua runtime. Never touch other
engines/eval scoring/runners. Bench write to benchmark_100seed_r78_prism only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices.
</default_follow_through_policy>
