<task>
r77-G1: MAKE BENCHMARK GRAPH GENERATION HASH-DETERMINISTIC -- oracle bug #5, highest
priority of the mop-up. The provenance probe (READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r76_REFS_PROVENANCE.md) proved
dagua/eval/graphs.py `_random_dag` builds edges via a string-keyed Python set before
from_edge_list, so node numbering/edge order (possibly topology) depend on PYTHONHASHSEED:
the benchmark graph is NOT reproducible across processes. Any per-seed comparison whose two
sides were generated in different processes on such graphs is permutation-corrupted.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-graph-determinism (branch
r77/graph-determinism, off develop -- CREATE IT: `git -C /home/jtaylor/projects/dagua
worktree add ~/.claude/worktrees/dagua-graph-determinism -b r77/graph-determinism develop`).
PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

WORK:
1. AUDIT dagua/eval/graphs.py (and any helpers it uses, incl dagua/graph.py
   from_edge_list) for EVERY hash-order dependence: sets/dicts of strings or tuples feeding
   edge lists or node orderings, iteration over set(...) without sorted(...), dict ordering
   that depends on insertion via set iteration. Enumerate ALL affected graph builders (not
   just _random_dag).
2. FIX: make generation fully deterministic (sorted iteration / ordered structures /
   explicit node index maps). CRITICAL DECISION -- canonical form: pick the DETERMINISTIC
   form closest to the graphs' documented intent (e.g. sorted string keys). This CHANGES
   the canonical graph realization for affected builders; that is acceptable and expected
   (the old realizations were process-dependent, i.e. not canonical at all). Do NOT try to
   reproduce any particular historical hash seed.
3. TEST: add a unit test that, for EVERY benchmark graph (get_test_graphs), spawns 2
   subprocesses with different PYTHONHASHSEED values, builds the graph in each, and asserts
   byte-identical edge_index/num_nodes/labels. This is the permanent tripwire.
4. ENUMERATE for the report: the list of affected graphs (those whose realization CHANGES
   under the fix vs an arbitrary current-process build -- test by comparing builds across
   hash seeds pre-fix) -- these graphs' benchmark rows (ALL engines, BOTH sides) need
   regeneration downstream; print the list loudly in the notes.

GATES (before commit): the new determinism test green (all graphs); pytest tests/ -k
"graphs or graph" -x -q green (fix any test that hard-coded a hash-dependent realization --
document each); ruff clean; the affected-graphs list enumerated. KNOWN pre-existing
failures (must not block): test_bench_large; classic_fcose; double-border smoke. Commit on
r77/graph-determinism. NO benches (the orchestrator handles regeneration).

DELIVERABLES: .project-context/research/sprint_rng_matching/r75_findings/
r77_GRAPH_DETERMINISM_NOTES.md (audit table: builder -> hash-dependent construct -> fix;
affected-graphs list; test evidence; commit sha). ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = every hash-order dependence in benchmark graph generation found+fixed, the
cross-hash-seed subprocess test green over ALL benchmark graphs, affected-graphs list
enumerated, committed. Partial audits are not acceptable -- the test IS the completeness
proof.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/graph-determinism only. Do NOT touch engine/pipeline code,
eval scoring, competitors, or reference runners -- graph GENERATION and its tests only.
No benches.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
