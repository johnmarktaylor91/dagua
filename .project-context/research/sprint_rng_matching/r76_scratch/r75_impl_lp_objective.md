<task>
Fix dagua's igraph-fidelity Sugiyama LP layering objective to match installed python-igraph 1.0.0.
An r75 probe PROVED dagua's current r74 objective is wrong: installed igraph 1.0.0 behaves like
its source quirk where BOTH degree vectors are filled with IGRAPH_IN, whereas dagua computes
out-strength for sources and in-strength for targets. Read the probe first:
.project-context/research/sprint_rng_matching/r75_findings/r75_PROBE_tails_RESULTS.md (section E1,
incl. the distinguishing DAG two_hubs_bridge: igraph y-layers [0,0,1,2,2,0,3,4,4] match the IN/IN
prediction, dagua's objective predicts [0,0,1,2,2,2,3,4,4]). Also read adversarial verdict 18 in
r75_ADVERSARIAL_VERDICTS.md (was NEEDS-EXPERIMENT; E1 is that experiment, now conclusive).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-lp-obj (branch r75/lp-objective). Work ONLY
here. Conventional commit (fix(sugiyama): ...). No push, no merge. NOTE: pre-commit runs
ruff-format -- if it reformats files the commit aborts; re-add and re-commit until it lands.

THE FIX:
- dagua/layout/ops/sugiyama.py:495-543 (_igraph LP objective coefficient computation; exact lines
  may have shifted after today's stage-A merge -- locate the function). Replace the out/in-strength
  objective with the faithful IN/IN semantics of igraph 1.0.0: replicate what
  _references/igraph/src/layout/sugiyama.c:588-615 does (both indegs and outdegs populated via
  IGRAPH_IN incidence, then the feedback-arc subtraction as written) -- port the BUGGY behavior
  faithfully; do not "correct" it.
- Gate: igraph-fidelity/default sugiyama modes ONLY (whatever paths feed the igraph LP layering);
  the graphviz path must be untouched (it ranks via network simplex, not this LP).
- Update the r74 objective assumption test (tests/test_layout/test_sugiyama_fidelity.py:87-105
  region) to the corrected expectation, and ADD a two_hubs_bridge-style regression test: build the
  probe's distinguishing DAG (edges from the E1 harness in /tmp/r75_probe.py if still present;
  otherwise reconstruct: it must be a DAG where IN/IN and out-in objectives yield different
  optimal layerings, verified against installed igraph at runtime in the test via
  ig.layout('sugiyama')). The test asserts dagua's layer assignment matches installed igraph's.

VERIFICATION LOOP:
a. pytest tests/ -k sugiyama -x -q -- all green.
b. Benchmark-path check (PYTHONPATH=$PWD, MPLCONFIGDIR=/tmp/mpl): run 5 seeds x
   {classic_sugiyama_default, classic_sugiyama_tight, classic_sugiyama_wide} on graphs
   binary_tree,densenet_block,real_karate_34,multiscale_skip_cascade via scripts/run_benchmark.py
   into /tmp/r75_lp_probe. Compare per-seed positions against the same run made from git stash
   (pre-change): report which graph/variant pairs CHANGED. Changes are EXPECTED only where the two
   objectives disagree; document each changed pair. Then compare changed layouts against the saved
   igraph references (main repo eval_output/benchmark_100seed_seeded_refs positions, read-only,
   igraph_sugiyama__for__* files; escalation_final as fallback): Procrustes/stress must MOVE TOWARD
   the reference (report numbers). If any changed pair moves AWAY from the reference, stop and
   diagnose before finishing.
c. Confirm graphviz-fidelity path unchanged: 2 seeds classic_sugiyama_graphviz_fidelity on
   binary_tree pre/post -- tensor-identical.
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_lp_objective_NOTES.md:
what changed, commit sha, per-pair before/after numbers, any pairs that moved away.
</task>
<completeness_contract>
Done = objective ported faithfully (IN/IN + feedback subtraction), gated, tests green incl. new
runtime-verified regression, probe shows changed pairs move toward the igraph reference,
graphviz path tensor-identical, notes written, committed.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/lp-objective only. Do not touch graphviz sugiyama paths, eval
metrics, or other engines.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
