<task>
r78-R2: execute the residual mop's OWN follow-up spec (READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r78_RESIDUAL_MOP.md -- it names
exactly what to do). Three items, generous runtime authorized (JMT: get clear answers):

A. SGD2 PAIR (2 rows, real_football_115 + wide_1_100_1 x classic_sgd2_multi_with_crossing):
implement the mop's scratch runner spec VERBATIM (same-process paired Dagua/reference,
seeds 100-199, tqdm disabled via env/monkeypatch, params exactly as specified, per-seed
hashes + stress + crossings persisted incrementally). Let it run as long as it takes
(hours are fine). Verdict per the spec: 100 paired seeds -> equivalence/identity verdict,
or a named first divergence in optimizer/loss sampling.

B. SFDP CLUSTER A (8 rows): build the instrumented pack.c dump the mop skipped
(/tmp/gv750-r2 via git archive of pinned 7.0.5; dump polyomino cells/margins/placement per
CC) on 2 representative graphs; diff vs dagua's packer; name the exact remaining
cell/margin/order difference OR prove parity (in which case the residual is elsewhere --
say where).

C. FDP (5 rows): the mop localized divergence BEFORE pack, in fdp_tLayout/fdp_xLayout.
Instrument those (same /tmp build): initial grid layout, force iterations, xLayout/prism
passes; first-divergence vs dagua's classic_fmmm_graphviz_fdp_fidelity path on
extreme_mixed_width_transformer + parallel_cycles_4x5 (1 seed). Name the first diverging
stage/quantity precisely.

If any named quantity is a PORTABLE rule, implement it gated to the relevant fidelity mode
(worktree: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-r2 -b r78/r2 develop`; gates: decisive improvement on target rows,
byte-identity samples on previously-identical family rows, family pytest green, ruff;
commit on r78/r2). Non-portable -> instrument-grade proof in the notes.

DELIVERABLES: append findings to r78_RESIDUAL_MOP.md (per-item verdicts w/ evidence tables,
ports+commit shas if any). ASCII. NO AI attribution. No push/merge. Clean /tmp scratch
(keep only small evidence excerpts).
</task>
<completeness_contract>
Done = all three items carry either a port (gates green) or instrument-grade evidence; the
sgd2 pair specifically must reach a real verdict (the compute is authorized) or name the
precise blocker beyond runtime.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/r2 only. NO runtime reference imports in dagua/layout.
Never touch other engines/eval scoring/runners.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices.
</default_follow_through_policy>
