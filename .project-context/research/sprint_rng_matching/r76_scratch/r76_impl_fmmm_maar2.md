<task>
r76-B2b: OGDF MAAR packing parity, attempt 2. The residual cluster GREW: after honest-ref
rescores, exactly 6 rows across TWO engines share this root cause -- fmmm 4 (random_dag_50
steps variants x3 + random_dag_200) and gem 2 (random_dag_50::classic_gem_iters2000,
random_dag_200::classic_gem_iters100). Both engines' component internals are proven matched
(fmmm RMSD 1e-3 vs honest runner; gem 5e-08 post-fix); ONLY the disconnected-components
packing (OGDF MAAR / TileToRowsCCPacker) tie-breaks diverge. Closing this closes the last
divergent rows for BOTH engines.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-fmmm-disc (branch r76/fmmm-disconnected,
exists from attempt 1). Work ONLY here. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

READ FIRST: r76_IMPL_fmmm_disc_NOTES.md (in
.project-context/research/sprint_rng_matching/r75_findings/ -- COMMITTED at 92b75b7 so it IS
in this worktree): attempt 1 tried 3 partial approaches (all reverted) against MAAR
singleton-packing tie-breaks and honestly parked. You start from its findings, NOT from
zero. Also read r75_findings/r76_PROBE_fmmm_triage.md (committed) for the leg/RMSD evidence.

METHOD -- INSTRUMENTED TRACE FIRST (this method just cracked two other engines today; do
NOT guess tie-break semantics): OGDF source at /home/jtaylor/tools/ogdf-src
(foxglove-202510) -- version-pin all cites there. The reference runner is
scripts/ogdf_runner (rebuilt r75, commit 0817427; source scripts/ogdf_runner.cpp). Build an
INSTRUMENTED local copy (copy runner source + patch to /tmp/ogdf-maar-trace; do NOT modify
the committed runner or ogdf-src) that dumps, for random_dag_50 + random_dag_200 (1-2
seeds): per-CC bounding boxes fed to the packer, the sort keys/order MAAR uses (perimeter?
width? height? area? insertion order for TIES -- singletons all have identical boxes, so
the tie ORDER is the whole game), per-row assignment decisions, and final CC offsets.
Compare against dagua's packer trace on the same inputs. Name the exact tie-break rule
(stable sort? original CC index? pointer order? -- find the comparator in
ogdf-src/src/ogdf/packing/TileToRowsCCPacker.cpp and its callers) and the first differing
placement decision.

THEN PORT: the named rule into dagua's OGDF-family packing path, gated so ONLY the
OGDF-family engines' disconnected path changes (fmmm/gem/(drl if shared)); graphviz-family
packers untouched.

GATES (all must pass before commit; else honest park -- this is the FINAL MAAR attempt):
1. The 6 rows' graphs: per-seed Procrustes RMSD vs honest references (r76_refs dir has them;
   MAIN repo read-only /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r76_refs)
   drops decisively on random_dag_50 + random_dag_200 for BOTH fmmm and gem (5 seeds each,
   params matched: steps10/100/200, iters100/2000).
2. Zero regressions: 5 previously-identical fmmm rows + 5 previously-identical gem rows
   (from MAIN repo eval_output/fidelity_definitive/r76_fmmm_rescore.jsonl and
   r76_gem_rescore.jsonl, quality_identical_raw=true; verify byte-identical dagua positions
   pre/post, 3 seeds each). Also 2 connected fmmm + 2 connected gem probe combos unchanged.
3. pytest tests/ -k "fmmm or gem" -x -q green; ruff clean.
KNOWN PRE-EXISTING FAILURE (verified on develop, NOT yours, must not block):
tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest.

DELIVERABLES: append "## Attempt 2: instrumented MAAR trace" to r76_IMPL_fmmm_disc_NOTES.md
(trace tables, the named tie-break rule w/ ogdf-src cites, before/after RMSD, gate evidence,
commit sha). Conventional commits on r76/fmmm-disconnected; re-add/re-commit through
ruff-format until `git log` SHOWS them. No push/merge. NO AI attribution. ASCII only. Clean
/tmp/ogdf-maar-trace at the end.
</task>
<completeness_contract>
Done = tie-break rule NAMED with trace evidence AND (gates 1-3 pass with commit, OR a
precise dossier showing the rule is non-portable [e.g. depends on heap pointer order] and
why, with NO commit -> rows go to aggregate tier). Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Never modify scripts/ogdf_runner (committed binary), ogdf-src, or
graphviz-family packers. Commits on r76/fmmm-disconnected only. Never modify files outside
the worktree except /tmp scratch. dagua ops never invoke the runner at runtime.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
