<task>
r77-M1: fix igraph disconnected-MDS component handling. The r77 thin-row probe (READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r76_THIN_ROW_DOSSIERS.md) proved
6 disconnected classical-MDS rows are NOT eigensolver-chaos floors: 1-ULP perturbation does
NOT reproduce the divergence; the difference lives in igraph's disconnected-MDS COMPONENT
HANDLING / packing versus dagua's. These rows carried false floor language since r75 --
JMT directive: perfect fidelity unless violence. Fix them.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mds-disc (branch r77/mds-disc, off
develop). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

CONTEXT: r75 shipped an igraph-faithful disconnected path (per-component MDS + literal DLA
merge port, ec24b05 + 8266c77, bit-identity 9/9 on ITS probe graphs) -- so the divergence
on THESE 6 rows is either a case the DLA port misses (component ordering? size ties?
placement-scale? seed threading?) or a different igraph code path for these graph shapes.
The 6 rows' graphs are listed in the dossier (read it).

STEP 1 -- FIRST-DIVERGENCE TRACE: pick 2 of the 6 graphs; run installed python-igraph
layout_mds on them (the reference adapter path in dagua/eval/competitors/
igraph_competitor.py) vs dagua's disconnected MDS, and bisect: per-component MDS outputs
first (should be identical per the r75 port -- verify), then the merge/packing decisions
(DLA placement order, radii, collision steps). Instrument dagua's DLA port; for igraph read
the C source (fetch matching source tarball to /tmp/igraph-src, READ ONLY) and, if source
reading is ambiguous, build an instrumented igraph in /tmp (fprintf in the layout_mds
merge path) -- the sanctioned trace-build pattern. Name the first differing decision.
STEP 2 -- PORT the named rule, gated to igraph-fidelity disconnected MDS.

GATES (before commit): the 6 rows' graphs: per-seed RMSD vs installed-igraph reference
drops decisively (5 seeds); zero regressions (byte-identical: the r75 bit-identity 9/9
probe set + 5 previously-identical mds rows, 3 seeds); pytest -k "mds" green; ruff clean.
KNOWN pre-existing failures (must not block): test_bench_large; classic_fcose;
double-border smoke. Commit on r77/mds-disc; re-bench the 6 graphs' classical_mds combos
into /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_mds (seeds 100-199,
--max-nodes 0, 0 errors).

DELIVERABLES: append "## M1 fix" to r76_THIN_ROW_DOSSIERS.md (trace tables, named rule w/
igraph source cite, before/after RMSD, gate evidence, commit sha). ASCII. NO AI
attribution. No push/merge. Clean /tmp scratch.
</task>
<completeness_contract>
Done = first divergence NAMED from trace AND (gated fix committed + clean bench, OR a
dossier proving the decision depends on unobservable state with the trace shown). The
prior floor label is already retracted -- an unfixed row without a new named cause is NOT
an acceptable endpoint.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/mds-disc only. Never touch other engines, eval scoring,
reference runners, installed igraph. Bench write to benchmark_100seed_r77_mds only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
