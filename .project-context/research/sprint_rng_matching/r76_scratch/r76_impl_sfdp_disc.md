<task>
r76-C4a: implement the Graphviz-faithful DISCONNECTED SFDP component loop + packing parity.
The r76 sfdp triage probe (READ FIRST:
.project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_sfdp_triage.md, Cluster A
+ recommendation ROI-1) identified this as the clearest sfdp fix: 21 divergent disconnected
rows across graphs parallel_cycles_4x5, random_dag_200, kitchen_sink_platform_graph,
multi_component_80, disconnected_encoder_residual, disconnected_label_cycle_collage,
random_bipartite_60, random_dag_50 -- spanning ALL THREE classic_sfdp variants (default,
graphviz_fidelity, p_neg2).

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sfdp-disc (branch r76/sfdp-disc, fresh off
develop). Work ONLY here. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

THE NAMED DIVERGENCE (from the probe, with pinned source):
- Graphviz 7.0.5 sfdp_layout(): ONE spring_electrical_control ctrl -> tuneControl() -> split
  via ccomps() -> for ncc>1, sfdpLayout(sg, ctrl, pad) per component REUSING THE SAME MUTABLE
  ctrl through the loop -> packSubgraphs(). Pin: `git -C
  /home/jtaylor/projects/_references/graphviz show 7.0.5:lib/sfdpgen/sfdpinit.c` (lines
  ~268-315) -- NEVER the working tree. Also read the pack machinery:
  7.0.5:lib/pack/pack.c (packSubgraphs/putGraphs/packing polyomino algorithm) as needed.
- Dagua: _layout_graphviz_sfdp_components() recursively runs layout_sfdp_pipeline per
  component with INDEPENDENT pipeline states, then _pack_component_positions() (the NEATO
  packer). Two divergence surfaces: (1) mutable ctrl state carried across components (field
  mutations from earlier components affect later ones -- trace which fields tuneControl/
  sfdpLayout mutate); (2) the packing algorithm itself (graphviz packSubgraphs vs dagua's
  neato-packer port -- offsets, margins, order, polyomino grid).
- Probe evidence to reproduce first: parallel_cycles_4x5 components [5,5,5,5] each identical
  local bbox [644.13, 672.97], packed bbox [1594.13, 1702.14] vs graphviz `-v` "pack info"
  flow; random_dag_200 has ~200 components (mostly singletons + one 181-node) with singleton
  local bboxes [0,0] -- check graphviz singleton placement (avg edge len=1.0 passes).

BLAST-RADIUS GUARDRAIL (r75 lesson, binding): shared packers are HIGH RISK -- r74's failure
mode was broad component/packing changes breaking already-good rows. If
_pack_component_positions (or any helper) is shared with neato/fmmm/fdp/other engines, DO NOT
change its default behavior -- add a gated parameter/new code path used only by classic_sfdp.
The fix may apply to all three classic_sfdp variants (all 21 rows span them) but must not
alter any other engine's output: verify by before/after position hashes on 2 neato + 2 fmmm
disconnected probe combos (5 seeds).

GATES (all must pass before commit; else document honestly, leave uncommitted):
1. Probe-path improvement: on >=4 of the 6 distinct disconnected graphs above, 5 seeds each,
   post-fix Procrustes RMSD vs reference materially improves (report before/after per graph;
   the probe's current medians are 0.07-0.38).
2. Zero regressions: previously-identical sfdp combos (pick 5 connected identical rows from
   eval_output/fidelity_definitive/r75_final.jsonl) keep byte-identical dagua positions
   pre/post fix (5 seeds); neato/fmmm disconnected probes unchanged (gate above).
3. pytest tests/ -k "sfdp" -x -q green; ruff clean on touched files.
4. NO runtime delegation: never invoke graphviz binaries from dagua ops at runtime (reference
   binaries are for offline probe comparison only).

DELIVERABLES:
.project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_sfdp_disc_NOTES.md
(what was ported w/ 7.0.5 cites, ctrl-field trace, packing parity evidence, before/after RMSD
tables, gate evidence, commit sha). Conventional commits on r76/sfdp-disc; re-add/re-commit
through ruff-format until `git log` SHOWS them. No push/merge. NO AI attribution in commits.
ASCII only.
</task>
<completeness_contract>
Done = gates 1-4 pass and committed, OR precise documented failure naming the exact unported
rule (7.0.5 cites) with NO commit. 2-attempt budget INSIDE this run (if your first approach
fails, one structured retry), then honest park. Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/sfdp-disc only. Never modify shared helper defaults used by
other engines; never touch eval scoring code, reference runners, or other engines' pipelines.
Never modify files outside the worktree.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
