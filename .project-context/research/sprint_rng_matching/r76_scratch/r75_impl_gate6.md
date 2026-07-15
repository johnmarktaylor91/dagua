<task>
Close the r74 "gate_6 loose end" in dagua's fidelity controls: the reference-self-split POSITIVE
control (gate_6) reads 0/scored in the tracked report because its control DATA was never
committed -- the control logic itself was validated separately in r74 (passes with
quality_identical_raw=True, battery_p_iut=7.96e-16). Regenerate the control data properly and
commit it so the tracked controls report shows gate_6 green.

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-gate6 (branch r75/gate6-data). Work ONLY
here. Conventional commit (fix(eval): ...). No push, no merge. Pre-commit ruff-format may
reformat; re-add and re-commit until the commit lands. NOTE the pre-commit large-file check:
if the control data exceeds it, store compactly (jsonl, no positions) -- read how other control
data in eval_output/fidelity_definitive/controls is stored and match that format/size.

CONTEXT TO READ FIRST:
- git log/show 0c07761 ("feat(eval): add reference-self-split positive control") -- how gate_6
  data is produced and consumed.
- scripts/definitive_fidelity_report.py -- gate_6 evaluation (search gate_6 /
  reference_self_split); what rows/files it expects in --controls-dir.
- The r74 note: control PASSES when generated (reference's own seeds split into two halves and
  battery-compared against itself -> must be quality-identical; a battery too weak to certify
  the reference against itself would be useless).
- eval_output/fidelity_definitive/controls/ in the MAIN repo (read-only) -- existing control
  data layout for gates 1-5.

DO:
1. Find or write the generation path for the reference-self-split control rows (there may be a
   flag/script from 0c07761; if generation requires reference positions, read them from the main
   repo's eval_output overlay dirs READ-ONLY -- e.g. benchmark_100seed_seeded_refs).
   Generate for a reasonable spread (the r74 validation used canonical seedable references;
   aim for ~20-40 control rows consistent with the other gates' scale).
2. Put the generated control data in the worktree's eval_output/fidelity_definitive/controls/
   (this path IS tracked for control data -- verify via git ls-files; if it is NOT tracked,
   determine where tracked control data lives and put it there; if truly nowhere is tracked,
   add the data dir + update the report's --controls-dir docs accordingly).
3. Run: python3 scripts/definitive_fidelity_report.py --controls --controls-dir <worktree
   controls dir> --output-dir /tmp/r75_gate6_check ; confirm gate_6 passed=true, scored>0,
   AND gate_5 still passed=true 0/40, gates 1/2/4 unchanged. gate_3 remains a known pre-existing
   failure -- do not touch it.
4. Commit the generation script changes (if any) + the control data + a short docs note.
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_gate6_NOTES.md:
how the data was generated (exact command), gate results before/after, commit sha.
</task>
<completeness_contract>
Done = gate_6 shows passed=true with scored>0 in a controls run against the committed data;
gates 1/2/4/5 unchanged; committed; notes written.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/gate6-data only. Main repo eval_output is READ-ONLY. Do not
modify gate logic or margins -- data generation only (plus a generation script if one is missing).
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
