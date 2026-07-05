<task>
r78-S2: SUPERIOR-DISTINCT EXHAUSTION -- JMT's fidelity bar is SAME, not better; 41 rows
where dagua beats the reference are still fidelity misses. For each: either PORT the
reference's worse behavior (gated to fidelity mode, exactly like every divergent-row port)
or PROVE the worse behavior is emergent/non-nameable. READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_fairness.md (the
fairness audit that vetted these rows) + the r77 ledger registry
(eval_output/fidelity_definitive_r77/OFFICIAL_R77_LEDGER.md; SUPERIOR_DISTINCT rows in
eval_output/fidelity_definitive/per_combo_r77.jsonl -- gem 24, fmmm 9, mds 8 by family).

WORKTREE: create `git -C /home/jtaylor/projects/dagua worktree add
~/.claude/worktrees/dagua-superior -b r78/superior develop`. PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl. NO runtime reference imports in dagua/layout (standing rule).

METHOD per family cluster (bisect-first, instrument-over-inference):
- GEM (24 rows): dagua and OGDF gem should be near-bit-exact after r76's round-budget fix
  -- superior-distinct here means our layout distribution lands in better basins. Bisect ONE
  representative (same-process both sides, instrumented runner from
  scripts/rng_match/build_ogdf_runner.sh recipe): first diverging update. If a nameable op
  (order/float path) -> port it (this may CONVERT the rows to identical). If pure float
  accumulation -> emergent proof w/ trace excerpt.
- FMMM (9 rows): same treatment.
- MDS (8 rows): the fairness probe tagged these as the eigenbasis class (reference's
  arbitrary basis pick lands on worse-stress planes). Verify with the M-chain method
  (eigengap + transform-within-eigenspace); if confirmed, they are the PROVEN
  equivalence-class case where the reference is a worse member -- document as terminal
  (matching would require the excluded LAPACK vendoring).

GATES: any port -> per-seed RMSD collapse on the ported rows + zero regressions
(byte-identity samples per family, 5 rows x 3 seeds) + pytest family tests green + ruff.
Every unported row -> instrument-grade emergent-behavior evidence (not r75-era inference).
KNOWN pre-existing failures (must not block): the standard 6-item list. Commit on
r78/superior; bench any ported family's affected graphs into
/home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r78_superior (seeds 100-199, 0
errors).

DELIVERABLES: .project-context/research/sprint_rng_matching/r75_findings/
r78_SUPERIOR_NOTES.md (per-cluster bisection, ports or proofs, before/after, gate evidence,
commit shas). ASCII. NO AI attribution. No push/merge.
</task>
<completeness_contract>
Done = all 41 rows either converted-to-identical (ported worse behavior), or carrying
instrument-grade emergent/equivalence-class proofs. "We're better" is not an endpoint
without the proof that matching requires an excluded measure.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r78/superior only. Never modify runners/eval scoring/other
engines. Bench write to benchmark_100seed_r78_superior only.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices.
</default_follow_through_policy>
