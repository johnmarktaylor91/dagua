<task>
r78-N1: the NEATO campaign -- the one engine family that never got the full treatment. 54
rows are divergent with only an r75 probe-level dossier ("exact CG/drand48/component
packing residual", r75_findings/r75_PROBE_tails_RESULTS.md -- READ IT plus the ledger
registry in eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md). JMT directive:
perfect fidelity unless violence. Apply the campaign-proven loop: bisect-first with an
instrumented reference build; port every named op; floor only with perturbation proof.

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-neato -b r78/neato develop`. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl. VERSION PIN: `git -C /home/jtaylor/projects/_references/graphviz
show 7.0.5:<path>` (lib/neatogen/*: neatoinit.c, stress.c/majorization, conjgrad.c,
distances via dijkstra/shortest paths, initial layout + drand48 seeding, disconnected
handling via pack). Reference runtime = installed neato 7.0.5.

METHOD: (1) pull the 54 rows from the ledger; cluster by graph class + failing legs (read
eval_output/fidelity_definitive/per_combo_r77.jsonl for their metrics; note 7 rows are
already MODE_B_IDENTICAL_DISTANCE and 15 close -- the neato reference may be deterministic
per seed: establish mode first); (2) instrumented graphviz build in /tmp/gv750-neato (git
archive; NEVER dirty the clone) dumping: initial coordinates + drand48 consumption, per-
iteration stress/CG residuals, termination, component pack decisions; (3) mirror from
dagua's neato pipeline; (4) first-divergence per cluster; (5) port gated to neato fidelity
mode; iterate. Same-process both-sides probes for any random_dag-class graph (hash lesson).

GATES (before commit): d_R (mode-B) or RMSD/W-gap improves decisively on >=2/3 of the
cluster representatives per ported op, with cumulative >=60% of the 54 rows materially
improved by campaign end OR each unimproved cluster carrying a named non-portable quantity
w/ dump; zero regressions (previously-identical neato rows byte-identical + the mode-B
identical-distance 7 stay); pytest -k neato green; ruff clean. KNOWN pre-existing failures
(must not block): the 6-item list in r76_scratch/r77_impl_a9b_guard.md. Commit on
r78/neato; bench the neato family <=300 into
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_neato (seeds 100-199,
0 errors).

DELIVERABLES: .project-context/research/sprint_rng_matching/r75_findings/
r78_IMPL_neato_NOTES.md (cluster table, bisection evidence per op, ports w/ 7.0.5 cites,
before/after, gate evidence, commit shas, bench line). ASCII. NO AI attribution. No
push/merge. Clean /tmp/gv750-neato.
</task>
<completeness_contract>
Done = every cluster of the 54 either ported (gates green) or floor/non-portable PROVEN
with instrument evidence. This family gets the same exhaustion standard as the rest of the
campaign -- an r75-era inference is not an endpoint.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/neato only. Never touch other engines, shared packer
defaults, eval scoring, reference runners. dagua never invokes neato at runtime. Bench
write to benchmark_100seed_r78_neato only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
