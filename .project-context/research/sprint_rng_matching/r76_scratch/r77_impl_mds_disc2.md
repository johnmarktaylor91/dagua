<task>
r77-M2: mds disconnected -- NATIVE port of the DLA walk-termination rule. The M1 attempt
(this worktree, commits bc72627+7c067bd) correctly TRACED the divergence but shipped a
FORBIDDEN fix: it delegates disconnected MDS to installed python-igraph at RUNTIME. HARD
RULE (standing, 4 prior incidents): dagua reimplementation engines must NEVER import or
invoke the reference package/binary at runtime -- the reference is for offline tracing
ONLY. The delegation will not be merged. Your job: keep M1's trace knowledge, REVERT the
delegation, and port the named rule natively.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mds-disc (branch r77/mds-disc, HEAD
7c067bd). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

WHAT M1 PROVED (read its M1 appendix in .project-context/research/sprint_rng_matching/
r75_findings/r76_THIN_ROW_DOSSIERS.md): per-component MDS already matches; the first
mismatch is at DLA merge entry -- dagua's DLA path consumes MORE rng draws than igraph
(random_dag_50: 1,050,140 vs 825,320; random_dag_200: 3,717,920 vs 3,052,376). The r75
"literal DLA merge port" (ec24b05) therefore diverges in its COLLISION/WALK-TERMINATION
rule (walks end earlier in igraph, or collision detection differs, or step proposals are
rejected differently).

WORK:
1. REVERT the runtime-delegation code from bc72627 (git revert or surgical removal --
   keep the M1 test only if rewritten to exercise the NATIVE path; keep the dossier).
2. DIFF THE DLA RULES: igraph's MDS disconnected merge source (C source to /tmp/igraph-src,
   READ ONLY; instrumented /tmp venv build if needed -- the sanctioned trace pattern) vs
   dagua's DLA port in dagua/layout/ops/ (find it via the r75 commit ec24b05). Instrument
   BOTH to dump per-walk: start position, step count, termination reason, collision cell,
   rng draws consumed. On random_dag_50 (1 seed), find the FIRST walk that differs and name
   the rule (termination radius? collision lookup? step distribution? draw order?).
3. PORT the named rule into dagua's DLA natively. Zero runtime igraph imports (AST check:
   no igraph import in dagua/layout/ -- add it to the tests).

GATES (before commit): per-seed RMSD vs installed-igraph reference drops decisively on the
6 M1 target graphs (5 seeds; generate reference OFFLINE via the eval adapter -- that is
the sanctioned path); rng draw counts MATCH igraph's on the traced probe; zero regressions
(r75 bit-identity 9/9 probe set + 5 previously-identical mds rows byte-identical); pytest
-k mds green; AST no-igraph-in-runtime test green; ruff clean. KNOWN pre-existing failures
(must not block): test_bench_large; classic_fcose; double-border smoke. Commit on
r77/mds-disc; re-bench the 6 graphs' classical_mds combos into
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_mds2 (seeds 100-199, 0
errors).
CAVEAT: random_dag_50/200 are affected by the hash-determinism bug (see
r75_findings/r76_REFS_PROVENANCE.md) -- generate reference AND dagua positions in the SAME
process for your probes (single script, both sides) so the graph realization is shared;
note this in the dossier.

DELIVERABLES: append "## M2: native DLA rule port" to r76_THIN_ROW_DOSSIERS.md (walk-diff
tables, the named rule w/ igraph source cite, draw-count parity, before/after RMSD, gate
evidence, commit shas). ASCII. NO AI attribution. No push/merge. Clean /tmp scratch.
</task>
<completeness_contract>
Done = delegation REVERTED + the DLA rule NAMED from walk-level trace AND (native port
committed with gates green + bench, OR a dossier proving the rule depends on igraph
internals that cannot be reproduced natively -- with the walk dump shown). Runtime
delegation is NEVER an acceptable endpoint.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/mds-disc only. NEVER import/invoke igraph from dagua
runtime modules (offline eval adapters excepted). Never modify installed igraph, other
engines, eval scoring. Bench write to benchmark_100seed_r77_mds2 only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
