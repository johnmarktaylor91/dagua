<task>
r77-A5: name and port the LAST x-coordinate constant for graphviz sugiyama -- the 1pt
half-width offset. JMT directive: perfect fidelity unless closing a gap requires violence;
this gap is a single traced constant away. Context (READ FIRST):
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_mincross_NOTES.md
sections A4b/A4c. A4c's endpoint: aux x-graph EDGE construction matches graphviz exactly,
but the LR-constraint minlens differ because dagua's ND_lw/ND_rw node half-widths are
~1pt-per-node SMALLER than graphviz's (gv 146,146,136 vs dagua 144,144,135 on
weighted_karate; 177,141 vs 175,139 on dense_pair_50).

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sugiyama-final (branch r77/sugiyama-final,
off develop which contains the FULL merged sugiyama stack). PYTHONPATH=$PWD;
MPLCONFIGDIR=/tmp/mpl.

STEP 1 -- NAME THE RULE: compare dagua's computed node boxes against `dot -Tjson` /
`dot -Tdot` ND_lw/ND_rw-derived values for the calibration graphs (weighted_karate_34,
dense_pair_50, hub_skip_superfan, heavy_tail_weights_50, binary_tree). The offset is
suspiciously uniform (~1pt per node -> 2pt per adjacent pair; one observed pair differs by
1). Candidate rules to check against pinned source (`git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:<path>`): penwidth/2 border
inflation, integer point rounding-up (ROUND/ceil in lib/common/geom.c or shapes.c
poly_inner/outer margins), GAP constant, label margin rounding. Name the exact source rule.

STEP 2 -- PORT it into the graphviz-fidelity node-box computation (the A4b helper in
dagua/eval/competitors/classic_competitor.py feeds engine inputs; the fix may belong there
or in the pipeline's box consumption -- follow where A4b put the box logic). Gated to
graphviz fidelity ONLY.

GATES (before commit):
a. Minlen parity: the A4c LR-minlen tables now match graphviz EXACTLY on weighted_karate_34
   and dense_pair_50 (146=146, 177=177 etc.).
b. Benchmark-path d_R improves on >=6 of a 10-row probe from the graphviz_fidelity
   close/far tiers (list from eval_output/fidelity_definitive/per_combo_r76.jsonl -- MAIN
   repo read-only; mode-B rows, d_R field), with NO row leaving the bit-exact/near tier.
c. Ordering discriminator unchanged 5/6; pytest tests/ -k "sugiyama or mincross or
   dot_rank" -x -q green; ruff clean. KNOWN pre-existing failures (must not block):
   test_bench_large hierarchy checkpoint; test_classic_competitor classic_fcose;
   test_cosmetic_node_features double-border render smoke.
d. Commit on r77/sugiyama-final (conventional, NO AI attribution). Then FULL family bench:
   run_benchmark --engines classic_sugiyama --variants --max-nodes 0 --seeds 100
   --seed-start 100 --workers 5 --timeout 3600 --watchdog-timeout 7200 --output-dir
   /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_sugiyama_a5
   -- Done line must show 0 errors.

DELIVERABLES: append "## A5: half-width rule" to r76_IMPL_mincross_NOTES.md (the named rule
w/ 7.0.5 cite, minlen parity tables, d_R before/after, gate evidence, commit sha, bench
Done line). ASCII only. No push/merge.
</task>
<completeness_contract>
Done = rule NAMED with source cite AND (gates pass with commit + clean family bench, OR a
dossier proving the offset is NOT a single rule with the trace evidence). The rule was
observed uniform across graphs -- "could not find it" is not an acceptable endpoint without
exhausting the candidate list above plus a direct shapes.c/geom.c read.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r77/sugiyama-final only. Never touch igraph/default paths,
eval scoring, reference runners. Bench write to benchmark_100seed_r77_sugiyama_a5 is the
only main-repo write allowed.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
