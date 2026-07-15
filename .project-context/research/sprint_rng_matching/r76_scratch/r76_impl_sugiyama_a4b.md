<task>
r76-A4b: graphviz sugiyama X-COORDINATE UNIT/NODE-BOX PARITY -- the final targeted attempt.
A4's Step-0 localization (READ FIRST, in this worktree:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md,
"## A4: rendered-stage parity") proved: ordering 5/6 exact (ba_500 internal mincross
graphviz 79098 vs port 79046); rendered real-node RANK AND ORDER MATCH on dense_pair_50,
weighted_karate_34, hub_skip_superfan; crossings diverge purely from X-COORDINATES. Named
defects (A4's own dossier -- your implementation spec):
1. NODE BOXES: dagua feeds its own node_sizes to the x-solver -- NARROWER than the graphviz
   reference boxes (hub_skip: dagua 44-47.85pt vs gv 54-61.56pt; dense_pair: 47.85-52.02 vs
   61.56-70.18; weighted_karate: 44 vs 54). For exact fidelity_mode="graphviz" the x-solver
   must consume the SAME boxes the reference dot computes for the benchmark DOT input.
   Determine how: inspect dagua/eval/competitors/graphviz_competitor.py::_graph_to_dot()
   -- if it emits explicit width/height/fixedsize, mirror those directly; if boxes come
   from label text metrics, mirror graphviz's box computation for the default
   shape/font/margins (validate against `dot -Tjson` boxes on the calibration graphs --
   exact match required, document the rule).
2. NODESEP UNITS: benchmark node_sep=1.0 maps to DOT nodesep=1.0 INCHES = 72pt; dagua's
   helper treated it as a layout unit. Port the units translation (position.c
   nodesep/ranksep consumption).
3. VIRTUAL-NODE WIDTHS + edge-pair auxiliary constraints per 7.0.5 position.c as needed.
A4 already tried nodesep-units ALONE: crossings improved 3/4 but stress toward-reference
only 2/4 -> reverted. Its conclusion: the box+units changes must land TOGETHER, validated
on stress AND crossings. That is this task.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mincross2 (branch r76/mincross). Uncommitted
WIP = the A1 ordering port (foundation -- do not revert). PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl. VERSION PIN: `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>` -- NEVER the working tree. Scope EVERYTHING to exact fidelity_mode="graphviz".

GATES (all must pass before commit; else document honestly, leave uncommitted):
a. Ordering discriminator unchanged: >=5/6 exact on the 6-graph calibration set.
b. Rendered crossings AND sampled stress BOTH move toward the reference on >=3/4 of
   {dense_pair_50, weighted_karate_34, hub_skip_superfan, heavy_tail_weights_50} (benchmark
   scoring path, 5 seeds; use A4's Step-0 local reference numbers as the baseline gaps) AND
   ba_500 rendered crossings gap to its local reference shrinks materially.
c. Zero regressions: 5 previously-identical sugiyama rows (igraph/default variants, from
   MAIN repo read-only /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/
   per_combo_r75.jsonl) keep byte-identical dagua positions pre/post (5 seeds).
d. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; pytest
   tests/test_layout/ -x -q green; ruff clean. KNOWN PRE-EXISTING FAILURE (verified on
   develop, NOT yours, must not block): tests/test_bench_large.py::
   test_hierarchy_checkpoint_rejects_incomplete_manifest.
e. ON PASS: commit the ENTIRE worktree stack as a logical series of conventional commits
   (A1 ordering port + weight gating + decompose order + A4b box/units parity + tests).
   Commits on r76/mincross AUTHORIZED and REQUIRED on pass; re-add/re-commit through
   ruff-format until `git log` SHOWS them. No push/merge. NO AI attribution.

DELIVERABLES: append "## A4b: x-coordinate box/units parity" to r76_IMPL_mincross_NOTES.md
(the box rule w/ validation vs dot -Tjson, units port cites, before/after crossings+stress
tables, gate evidence, commit shas). ASCII only.
</task>
<completeness_contract>
Done = gates a-e pass with the full stack committed, OR a precise dossier naming exactly
which x-constraint remains unmatched (with 7.0.5 position.c cites + measured residual) and
NO commit -- that dossier becomes the family's official port-in-progress disposition. Never
weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/mincross only. Never touch igraph/default paths, BK, eval
scoring code, reference runners. Never modify files outside the worktree. dagua ops never
invoke graphviz at runtime.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
