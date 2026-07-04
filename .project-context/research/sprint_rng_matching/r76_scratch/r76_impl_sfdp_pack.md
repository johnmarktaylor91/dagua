<task>
r76-C4d: port graphviz packSubgraphs POLYOMINO PACKING for disconnected SFDP -- the final
sfdp work item. The shared-RNG-stream fix landed (develop; RMSD improved 6/6 disconnected
graphs) but 5 disconnected graph clusters remain statistically divergent with the packing
GEOMETRY as the named residual cause, and on 2 of them dagua's quality is slightly WORSE
than the reference (fails the sprint's quality-parity bar), so a faithful port is required
before any disposition:

| graph | dagua W | ref W (fresh refs, seeds 100-199) |
|---|---:|---:|
| disconnected_encoder_residual | 0.7636 | 0.7688 (parity) |
| disconnected_label_cycle_collage | 0.5110 | 0.7634 (dagua better) |
| kitchen_sink_platform_graph | 0.4997 | 0.4791 (dagua 4.4% WORSE) |
| multi_component_80 | 0.9078 | 0.8662 (dagua 4.8% WORSE) |
| random_dag_50 | 1.0971 | 1.1204 (dagua better) |

WORKTREE: /home/jtaylor/.claude/worktrees/dagua-sfdp-pack (branch r76/sfdp-pack, fresh off
develop -- develop already contains the RNG-stream + CSR-order + unit-weights sfdp fixes).
Work ONLY here. PYTHONPATH=$PWD; MPLCONFIGDIR=/tmp/mpl.

READ FIRST: .project-context/research/sprint_rng_matching/r75_findings/
r76_IMPL_sfdp_disc_NOTES.md (the RNG fix + its source trace of the pack call path) and
r76_PROBE_sfdp_triage.md (Cluster A). Pinned source: `git -C
/home/jtaylor/projects/_references/graphviz show 7.0.5:lib/pack/pack.c` (packSubgraphs ->
packGraphs -> putGraphs -> polyGraphs: per-CC polyomino cell generation from node boxes --
and note pinfo.doSplines=1 set in sfdpinit.c means SPLINE boxes are included in polyomino
occupancy; sfdp routes splines BEFORE packing), placement loop (sort key + tie handling,
placeGraph grid search order), CL_OFFSET margins.

STEP 1 -- INSTRUMENTED TRACE (mandatory; the method that cracked 3 engines this sprint):
`mkdir -p /tmp/gv750-pack && git -C /home/jtaylor/projects/_references/graphviz archive
7.0.5 | tar -x -C /tmp/gv750-pack` (NEVER dirty the reference clone). Instrument pack.c to
dump per-CC: bbox, polyomino cell set size, sort key + final sort order, placement (x,y)
decisions in sequence. Trace multi_component_80 + kitchen_sink_platform_graph (the two
quality-worse graphs), seed 100, DOT via dagua/eval/competitors/graphviz_competitor.py.
Compare against dagua's packer trace on identical inputs; name the first differing
placement decision and its rule.

STEP 2 -- PORT into dagua's sfdp disconnected path ONLY (the r75 guardrail stands: never
change shared packer defaults used by neato/fmmm/fdp -- new gated code path). The doSplines
occupancy question: dagua has no splines at pack time; if spline-box occupancy materially
changes polyomino shapes, approximate with edge straight-line boxes and MEASURE whether
that closes the gap; document honestly if spline-box parity is the irreducible residual.

GATES (all before commit; else honest dossier -- this is the FINAL sfdp item):
1. Benchmark-path W-stress gap to reference shrinks materially on >=4/5 graphs above
   (5 seeds, params matched, fresh refs at
   /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r76_sfdp_refs read-only);
   kitchen_sink + multi_component must reach parity-or-better OR the dossier must name
   exactly which packing rule cannot be reproduced and why.
2. Zero regressions: 5 previously-identical sfdp rows byte-identical pre/post (5 seeds);
   2 neato + 2 fmmm disconnected combos unchanged position hashes; connected sfdp rows
   byte-identical (5 sample rows).
3. pytest tests/ -k "sfdp" -x -q green; ruff clean. KNOWN PRE-EXISTING FAILURES (NOT yours,
   must not block): tests/test_bench_large.py::test_hierarchy_checkpoint_rejects_incomplete_manifest,
   tests/test_classic_competitor.py::test_classic_competitor_names_match_expected_values.

DELIVERABLES: append "## Packing parity (C4d)" to r76_IMPL_sfdp_disc_NOTES.md (trace tables,
named rules w/ pack.c cites, before/after W table, gate evidence, commit sha). Conventional
commits on r76/sfdp-pack; re-add/re-commit through ruff-format until `git log` SHOWS them.
No push/merge. NO AI attribution. ASCII only. Clean /tmp/gv750-pack at the end.
</task>
<completeness_contract>
Done = gates 1-3 pass and committed, OR the step-1 trace + a dossier naming the exact
non-reproducible packing rule (e.g. spline-box occupancy) with measured residual impact and
NO commit. Never weaken a gate.
</completeness_contract>
<action_safety>
No pushes/merges. Never modify shared packer defaults, neato/fmmm/fdp paths, eval scoring,
reference runners, or /home/jtaylor/projects/_references/graphviz. Commits on r76/sfdp-pack
only. Never modify files outside the worktree except /tmp/gv750-pack scratch.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
