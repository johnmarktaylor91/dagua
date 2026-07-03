<task>
r76-A4c: port graphviz's AUXILIARY X-GRAPH construction -- the FINAL sugiyama push of the
sprint. The A4b dossier (READ FIRST, in this worktree:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md,
"## A4b" section) landed box/units parity and got the calibration set to:
weighted_karate_34 crossings 108/108 EXACT (stress 0.634794 vs 0.634682), hub_skip_superfan
2/2 (stress 0.5436 vs 0.5400), dense_pair_50 343 vs 331, heavy_tail_weights_50 63 vs 67 --
but "toward-reference" moved on only 2/4, and A4b NAMED the residual: the graphviz
position.c AUXILIARY X-GRAPH -- make_edge_pairs(), LR edge generation, slack/aux virtual
nodes, their INITIAL RANKS, and node/edge INSERTION ORDER (saved fast-edge traversal) --
which dagua's x-network-simplex does not yet mirror.

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-mincross2 (branch r76/mincross, HEAD
525327d = the COMMITTED A1 ordering port + A4b box/units stack -- your foundation; add
commits on top, never reset). PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

THE WORK:
1. TRACE the reference aux graph: `git -C /home/jtaylor/projects/_references/graphviz show
   7.0.5:lib/dotgen/position.c` (make_X* / make_edge_pairs / allocation order),
   7.0.5:lib/common/ns.c (how the aux graph's ranks are solved). If source reading is
   ambiguous, instrument: `mkdir -p /tmp/gv750-a4c && git -C
   /home/jtaylor/projects/_references/graphviz archive 7.0.5 | tar -x -C /tmp/gv750-a4c`,
   add fprintf dumps of the aux-graph node list, edge list (tail/head/weight/minlen) in
   creation order, initial ranks, and final xcoords for weighted_karate_34 + dense_pair_50
   (DOT built exactly as dagua/eval/competitors/graphviz_competitor.py; NEVER dirty the
   reference clone; clean /tmp/gv750-a4c at the end).
2. MIRROR it in dagua's graphviz-fidelity x-stage: same aux node/edge construction, same
   weights/minlens (omega tables), same insertion order, same initial-rank seeding, gated to
   exact fidelity_mode="graphviz" ONLY.
3. REVIEW A4b's classic_competitor.py change (the DOT node-box helper w/ Helvetica width
   tables): engine-INPUT plumbing (what node_sizes the graphviz-fidelity engine receives) is
   acceptable; anything leaking into SCORING or other engines' inputs is not -- verify it is
   gated to graphviz-fidelity engine variants, relocate/gate if not, and justify in the
   notes either way.

GATES (all must pass before final commits; else document honestly -- the A4b+A4c dossier
then becomes the family's official port-in-progress disposition and this line of work ENDS
for the sprint):
a. Ordering discriminator unchanged: >=5/6 exact.
b. Rendered crossings AND stress move toward reference on >=3/4 of {dense_pair_50,
   weighted_karate_34, hub_skip_superfan, heavy_tail_weights_50} vs the A4b table above
   (exact matches count as "toward"); report the full table.
c. Zero regressions: 5 previously-identical sugiyama rows (igraph/default variants, MAIN
   repo read-only /home/jtaylor/projects/dagua/eval_output/fidelity_definitive/
   per_combo_r75.jsonl) byte-identical dagua positions pre/post (5 seeds).
d. pytest tests/ -k "sugiyama or mincross or dot_rank" -x -q green; pytest
   tests/test_layout/ -x -q green; ruff clean. KNOWN PRE-EXISTING FAILURES (verified on
   develop, NOT yours, must not block): tests/test_bench_large.py::
   test_hierarchy_checkpoint_rejects_incomplete_manifest AND tests/test_classic_competitor.py::
   test_classic_competitor_names_match_expected_values (classic_fcose name drift).
e. ON PASS: conventional commits on r76/mincross (the wip commit 525327d may be followed by
   clean commits; do NOT rewrite history). NO AI attribution. No push/merge.

DELIVERABLES: append "## A4c: auxiliary x-graph parity" to r76_IMPL_mincross_NOTES.md
(aux-graph trace tables port-vs-reference, what was ported w/ position.c/ns.c cites,
before/after calibration table, competitor-helper review verdict, gate evidence, commit
shas). ASCII only.
</task>
<completeness_contract>
Done = gates a-e pass and committed, OR a dossier naming exactly which aux-graph element
still mismatches (with trace evidence + 7.0.5 cites) and NO new commit beyond the protective
wip -- that dossier is the official end-state for graphviz-sugiyama this sprint. Never weaken
a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/mincross only. Never touch igraph/default paths, BK, eval
SCORING code, reference runners. classic_competitor.py may ONLY carry engine-input plumbing
gated to graphviz-fidelity (review mandate above). Never modify files outside the worktree
except /tmp/gv750-a4c scratch.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
