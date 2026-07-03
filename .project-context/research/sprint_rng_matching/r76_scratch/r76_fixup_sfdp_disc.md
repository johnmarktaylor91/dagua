<task>
r76-C4a-FIXUP: the ONE sanctioned focused fixup for disconnected SFDP -- port graphviz's
SINGLE CONTINUOUS rand() STREAM across the component loop. The prior run (2 attempts, both
reverted -- READ FIRST, both files ARE in this worktree now:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_sfdp_disc_NOTES.md and
r76_PROBE_sfdp_triage.md) failed its gate but NAMED the exact unported rule it never
implemented:

THE RULE: graphviz runs ONE process-wide C rand() stream through the ENTIRE disconnected
loop. Per component inside sfdpLayout(): Multilevel.c coarsening consumes random_permutation()
draws FIRST; then spring_electrical.c reseeds ONLY `if (ctrl->random_start)
srand(ctrl->random_seed)` before random initial positions; prolongation then sets
ctrl->random_start=FALSE (plus K*=0.75, adaptive_cooling=FALSE, step changes) ON THE SHARED
ctrl. So component order, prior components' draw counts, and the mutated random_start all
determine later components' streams. Dagua currently gives each component an independent
GraphvizRandom -- that is the divergence. Prior attempts only carried ctrl scalar state
(their mutation-timing findings are in the notes -- reuse them); neither threaded the stream.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sfdp-disc (branch r76/sfdp-disc, clean --
attempts reverted). Work ONLY here. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

STEP 1 -- GROUND TRUTH VIA INSTRUMENTED TRACE BUILD (mandatory; do NOT guess the semantics):
Extract graphviz 7.0.5 source to your own scratch: `git -C
/home/jtaylor/projects/_references/graphviz archive 7.0.5 | tar -x -C /tmp/gv750-disc`
(mkdir first; NEVER dirty the reference clone; /tmp/gv750-trace belongs to a PARALLEL task --
do not touch it). Instrument with fprintf(stderr,...): (a) a draw counter around rand()
consumers (random_permutation in lib/sfdpgen/Multilevel.c, initial-position drand/rand in
lib/sfdpgen/spring_electrical.c, prolongation jitter), (b) srand() call sites with the seed
value and current random_start, (c) per-component boundaries in lib/sfdpgen/sfdpinit.c
(268-315) with component size + whether sfdpinit resets ctrl fields per component. Build just
enough to run `sfdp` (or a tiny C harness calling sfdp_layout) and trace TWO graphs:
parallel_cycles_4x5 and multi_component_80 (DOT built exactly as
dagua/eval/competitors/graphviz_competitor.py does; seed via the same mechanism the benchmark
uses). Deliverable of this step: a per-component table -- draws consumed in coarsening /
reseed fired? / draws in init / draws in prolongation -- that pins the exact stream schedule.

STEP 2 -- PORT: thread ONE GraphvizRandom instance through dagua's disconnected SFDP loop
reproducing that schedule exactly (consumption order, conditional reseed semantics, ctrl
mutation timing from the notes), gated so ONLY classic_sfdp disconnected behavior changes.
Shared-helper guardrail unchanged: never alter defaults used by neato/fmmm/fdp -- new
parameters/paths only.

GATES (all must pass before commit; else document honestly, leave uncommitted -- this is the
FINAL disc attempt, park after):
1. Focused benchmark (same command shape as the notes: 6 disconnected graphs x 3 variants x
   5 seeds, --seed-refs graphviz_sfdp): median Procrustes RMSD improves on >=4 of 6 graphs;
   with a correct stream port expect DRAMATIC drops on at least the pure-random-layout
   graphs. Report the full before/after table.
2. Zero regressions: 5 previously-identical connected sfdp rows (list from the MAIN repo,
   read-only: /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/r75_final.jsonl,
   engine contains sfdp, quality_identical_raw=true) keep byte-identical dagua positions
   pre/post (5 seeds); 2 neato + 2 fmmm disconnected probe combos unchanged position hashes.
3. pytest tests/ -k "sfdp" -x -q green; ruff clean on touched files.
4. No runtime delegation (dagua never invokes graphviz at runtime).

DELIVERABLES: append "## Fixup (attempt 3): shared RNG stream" to r76_IMPL_sfdp_disc_NOTES.md
(the step-1 stream schedule table, what was ported w/ 7.0.5 cites, before/after RMSD table,
gate evidence, commit sha). Conventional commits on r76/sfdp-disc; re-add/re-commit through
ruff-format until `git log` SHOWS them. Commits on this branch are AUTHORIZED and required on
gate pass. No push/merge. NO AI attribution in commits. ASCII only. Clean up /tmp/gv750-disc
at the end.
</task>
<completeness_contract>
Done = gates 1-4 pass and committed, OR the step-1 trace table + a precise statement of which
schedule element cannot be reproduced and why, with NO commit. Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Never write to /home/jtaylor/projects/_references/graphviz or
/tmp/gv750-trace. Never modify shared helper defaults used by other engines; never touch eval
scoring code or reference runners. Never modify files outside this worktree except
/tmp/gv750-disc scratch.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
