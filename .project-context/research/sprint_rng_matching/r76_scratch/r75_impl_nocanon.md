<task>
Implement two APPROVED scoring/report changes for dagua's fidelity system (JMT signed off today).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-nocanon (branch r75/no-canon-tier). Work ONLY
here. Conventional commit (feat(eval): ...). No push, no merge. Pre-commit runs ruff-format; if it
reformats, re-add and re-commit until the commit lands.

CHANGE 1 -- "no-canonical-reference" tier for reference-inexpressible variants:
Background (documented in .project-context/research/sprint_rng_matching/r75_findings/
r75_sfdp_codex.md finding 1 + r75_ADVERSARIAL_VERDICTS.md verdict 1): graphviz 7.0.5 sfdp does
NOT read `theta` or `maxiter` graph attributes (theta is the compile-time constant bh=0.6 in
7.0.5:lib/sfdpgen/spring_electrical.c; verify via `git -C /home/jtaylor/projects/_references/
graphviz show 7.0.5:<path>` -- the working tree is a NEWER version). The benchmark passed those
attrs and graphviz silently ignored them (empirically proven: reference positions bit-identical
across settings, rms 4e-16). Therefore variants classic_sfdp_theta04, classic_sfdp_theta08,
classic_sfdp_steps200 have NO expressible canonical reference: fidelity is not a coherent concept
for them (JMT: "graphviz cant make em in the first place so fidelity doesnt make sense").
Implement:
- dagua/eval/variants.py: add a machine-readable flag on the variant record (e.g.
  reference_expressible: bool = True, set False for these 3) with a docstring/comment explaining
  the 7.0.5 evidence (cite spring_electrical.c constant + the probe result). Keep the variants
  themselves working (they are legitimate dagua extension knobs).
- scripts/definitive_fidelity_analysis.py: rows for reference-inexpressible variants get a
  persisted field no_canonical_reference: true and are routed OUT of divergent/identical
  accounting entirely (their own tier, analogous to the exploratory tier from r74 -- find how
  quality_identical_exploratory rows are excluded and mirror that pattern).
- scripts/definitive_fidelity_report.py: report these rows as a separate, clearly-documented
  section/line: "NO CANONICAL REFERENCE (dagua extension parameters -- reference cannot express
  these settings; excluded from fidelity accounting)" with the per-variant counts and a short
  rationale paragraph citing the evidence. They must NOT appear in divergent counts, identical
  headlines, or the north-star denominator.
- Tests: fixture rows for a flagged variant -> assert routing to the new tier, exclusion from
  headline/divergent counts, presence in the report section. Extend
  tests/test_quality_battery_correctness.py or the report tests as appropriate.

CHANGE 2 -- QUALITY_SUPERIOR_DISTINCT scorecard line (flag already persisted):
The analysis already persists quality_superior_distinct (landed today in commit 205129e --
read its implementation in scripts/definitive_fidelity_analysis.py). Add to
scripts/definitive_fidelity_report.py a separate scorecard line/section: "QUALITY-SUPERIOR BUT
DISTINCT (dagua measurably better on every failing quality leg -- these layouts are DIFFERENT
from the reference, not equivalent)" listing count + per-engine breakdown. JMT's requirement
verbatim: "make clear they are in fact different." These rows remain counted as
divergent/non-identical in all fidelity accounting -- the line is informational overlay, not a
reclassification. Test: fixture asserting a quality_superior_distinct row still counts divergent
AND appears in the new section.

VERIFICATION:
a. pytest tests/ -k "quality or battery or report" -x -q green (never weaken existing asserts).
b. Controls: gate_5 must stay 0/40 -- run scripts/definitive_fidelity_report.py --controls
   --controls-dir /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/controls
   --output-dir /tmp/r75_nocanon_controls (gate_3/gate_6 failures are pre-existing; only gate_5
   regression blocks you).
c. Dry-run the report against the existing rescore data (read-only from main repo:
   eval_output/fidelity_definitive/r75_truebaseline.jsonl or r74_phase2_rescore.jsonl -- check
   which the report can consume with --per-combo) and paste the new sections' rendered output in
   your notes, with before/after divergent totals demonstrating exactly 47 rows moved to the
   no-canonical tier and 0 rows changed identity tiers.
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_nocanon_NOTES.md:
what changed, commit sha, rendered report excerpts, before/after counts, controls result.
</task>
<completeness_contract>
Done = both changes implemented + tested + committed, dry-run shows exactly the 47 expected rows
in the new tier and zero identity-tier changes, gate_5 0/40 held, notes written.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/no-canon-tier only. Do NOT touch layout pipeline code or
dagua/metrics.py. Main repo is read-only reference data.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
