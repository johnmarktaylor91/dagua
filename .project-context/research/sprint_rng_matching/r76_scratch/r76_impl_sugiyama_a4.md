<task>
r76-A4: graphviz sugiyama RENDERED-STAGE parity (stages B-D) on top of the completed
ordering-stage port. This is the last major sugiyama work item of a two-month fidelity
campaign (~50 graphviz-family rows hinge on it).

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mincross2 (branch r76/mincross). It contains
UNCOMMITTED WIP = the A1 mincross ordering port (4 modified files + notes). That WIP is your
FOUNDATION -- do not revert it. READ FIRST (in worktree):
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md.
A1 status: ordering-stage discriminator 5/6 EXACT vs `dot -v` (binary_tree 0, bipartite 36,
hub_skip 2, weighted_karate 178->63 exact, dense_pair 271; heavy_tail_weights_50 off by ONE:
pass-0 91 vs 96, final 51 vs 50 -- residual named "expanded fast-edge metadata/order",
DO NOT chase it unless step-0 shows it drives rendered divergence). Perf: ba_500 mincross
33.7s (incremental transpose landed). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

CONTEXT -- WHY RENDERED != ORDERING: prior ladder saw rendered crossings move toward
reference on only 2/4 calibration graphs, and ba_500 rendered ~95,261 vs REFERENCE dot
~140,276 (old dagua: ~117,932). NOTE THE FRAME: this is a FIDELITY campaign -- the target is
MATCHING the reference's rendered output (crossings/stress distributions), NEVER "fewer
crossings than dot". Dagua rendering FEWER crossings than dot is a FAILURE here. The
divergence between exact ordering parity and rendered mismatch lives in the post-ordering
stages: graphviz position.c x-coordinates (stage-A xns port landed in r75 -- verify it is
actually active for fidelity_mode="graphviz" in the benchmark path), flat-edge handling
(mincross.c flat_* machinery), same-rank/multi-edge/self-loop treatment, and any
order-to-coordinate plumbing gaps in dagua's pipeline.

STEP 0 -- LOCALIZE (mandatory before porting anything): for ba_500 + the calibration graphs
{dense_pair_50, weighted_karate_34, hub_skip_superfan, heavy_tail_weights_50}, run BOTH:
(i) the ordering discriminator (`dot -v` mincross count vs port count -- ba_500 is now fast
enough), and (ii) the benchmark-path rendered crossings for dagua vs the reference (generate
reference positions via dagua/eval/competitors/graphviz_competitor.py -- the sanctioned
offline reference path; note ba_500 crossings are SAMPLED estimates, use the scoring module's
estimator + report SE, seeds 100-102 suffice for localization). Build a per-graph table:
ordering-gap vs rendered-gap. Every graph where ordering matches but rendered diverges names
a DOWNSTREAM defect; trace WHERE dagua's rendered order/coords depart from the mincross
ordering (is the final node order actually the mincross order? are x-coords collapsing/
reordering nodes within ranks? are dummy chains rendered as bends vs straight?).

STEP 1 -- PORT the named downstream defects from pinned graphviz 7.0.5 (`git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:<path>` -- NEVER the working tree):
lib/dotgen/position.c (x-coord priority/medians if the r75 xns stage is incomplete/inactive),
lib/dotgen/mincross.c flat-edge machinery, lib/dotgen/dotsplines.c ONLY insofar as node
positions are affected (dagua scores node-position-derived metrics, not spline geometry).
Scope EVERYTHING to exact fidelity_mode="graphviz"; igraph/default/dot-alias paths untouched.

GATES (all must pass before commit; else document honestly, leave uncommitted):
a. Ordering discriminator stays >=5/6 exact on the 6-graph calibration set (no regression of
   the A1 result).
b. Rendered crossings AND stress move TOWARD the reference on >=3/4 calibration graphs
   (benchmark scoring path, 5 seeds), with the movement material (>25% gap reduction), and
   ba_500::graphviz_fidelity rendered crossings move toward the reference dot value
   (report before/after gaps with SE).
c. Zero regressions: 5 previously-identical sugiyama rows (igraph/default variants; list
   from MAIN repo read-only
   /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/per_combo_r75.jsonl --
   engine contains sugiyama, quality_identical_raw=true) keep byte-identical dagua positions
   pre/post (5 seeds).
d. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; pytest
   tests/test_layout/ -x -q green; ruff clean. (Known pre-existing failure NOT yours:
   tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest --
   verify it also fails on the base commit with `git stash` or a scratch checkout, note the
   verification, then ignore it.)
e. ON GATES PASSING: commit the ENTIRE worktree stack as a logical series of conventional
   commits (the A1 ordering port + weight gating + decompose order + your A4 stage work +
   tests). Commits on r76/mincross are AUTHORIZED and REQUIRED on pass; re-add/re-commit
   through ruff-format until `git log` SHOWS them. No push/merge. NO AI attribution.

DELIVERABLES: append "## A4: rendered-stage parity" to r76_IMPL_mincross_NOTES.md (step-0
localization table, what was ported w/ 7.0.5 cites, before/after gap tables, gate evidence,
commit shas). ASCII only.
</task>
<completeness_contract>
Done = gates a-e pass with the full stack committed, OR the step-0 localization table + a
precise dossier naming which downstream stage remains unported and its 7.0.5 cite, with NO
commit. An honest documented failure is acceptable; a false pass is not. Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/mincross only. Never touch igraph/default ordering, BK
paths, eval scoring code, reference runners. Never modify files outside the worktree. dagua
ops never invoke graphviz at runtime (competitor-based reference generation in eval/ is the
sanctioned offline path).
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
