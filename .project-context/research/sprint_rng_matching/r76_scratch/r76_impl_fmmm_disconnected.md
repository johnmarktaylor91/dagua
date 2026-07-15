<task>
r76-B2: fix dagua's OGDF-FMMM fidelity parity for DISCONNECTED graphs. Probe evidence
(.project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_fmmm_triage.md -- READ IT
FIRST): connected graphs already match the honest runner at RMSD 1e-17..1e-3, but every
disconnected combo is structurally apart at RMSD ~0.05-0.10 AT ALL STEP COUNTS (not chaotic
growth): random_dag_50 (0.099/0.089/0.087 at steps10/100/200), random_dag_200 (0.051),
multi_component_80 (0.090), kitchen_sink_platform_graph (0.057). The RNG primitive is proven
bit-exact (_OgdfMt19937 matches libstdc++ mt19937+uniform_int_distribution first 20 draws).
The defect is in COMPONENT HANDLING: ordering, per-component RNG consumption, packing offsets.

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-fmmm-disc (branch r76/fmmm-disconnected).
Work ONLY here; PYTHONPATH=$PWD. Conventional commit (fix(fmmm): ...); pre-commit ruff-format may
reformat -- re-add and re-commit until `git log` SHOWS the commit. No push/merge.

REFERENCE (the runner's actual source): /home/jtaylor/tools/ogdf-src (foxglove-202510):
- src/ogdf/energybased/FMMMLayout.cpp -- call(), the connected-component split
  (createInducedSubgraphs / componentSplitterLayout path? READ what FMMM actually does for
  disconnected input: search for connected components handling, pack routines
  (calculate_bounding_rectangles_of_components, rotate/pack), and WHERE randomNumber draws occur
  relative to per-component processing (one global stream across components in discovery order).
- The runner (scripts/ogdf_runner.cpp in the MAIN repo, read-only) seeds once; components consume
  one continuous stream.
DAGUA: dagua/layout/ops/pipelines/fmmm.py (_layout_fmmm_fidelity_components at ~:1787-1825 --
component split, per-component pipeline calls, packing) + ops/fmmm.py.

METHOD (trace-first, do not guess):
1. Instrument BOTH sides on random_dag_50 seed 100: dump per-component node lists in processing
   order, per-component RNG draw counts, initial placements, and final per-component bboxes.
   For OGDF: build a /tmp instrumented runner from a PATCHED COPY of the needed sources -- follow
   the pristine-restore contract: `git -C /home/jtaylor/tools/ogdf-src status` must be clean at
   the end, and rebuild scripts/ogdf_runner... NO -- do NOT touch scripts/ogdf_runner or the
   ogdf-src tree state permanently: build your instrumented binary to /tmp/ogdf_runner_dump2
   using a separate cmake build dir (/tmp/ogdf-build-fmmm), then `git checkout -- .` the source
   tree and VERIFY `git status` clean. Do NOT rebuild or replace scripts/ogdf_runner (it is
   correct and freshly committed).
2. Find the first divergence: component order? RNG draws consumed per component? packing
   rectangle order/rotation? zero-origin recentering?
3. Fix dagua's disconnected path to mirror OGDF exactly (single RNG stream in OGDF's component
   order, identical packing math). Gate: fidelity_mode only; connected path BIT-IDENTICAL
   pre/post (torch.equal, 3 connected graphs x 3 seeds -- e.g. deep_chain_20, grid_5x5,
   tl_mlp_3layer).
VERIFICATION:
a. random_dag_50 steps10 seeds 100-104 via benchmark path: RMSD vs /tmp instrumented-or-plain
   current runner drops from ~0.099 to <0.01 (report per-seed numbers; <0.001 = excellent).
b. multi_component_80 + kitchen_sink_platform_graph + random_dag_200 steps10, 3 seeds: RMSD
   improves materially on all (report before/after).
c. Connected bit-identity gate + pytest tests/ -k fmmm -x -q green.
Write .project-context/research/sprint_rng_matching/r75_findings/r76_IMPL_fmmm_disc_NOTES.md:
first-divergence finding, fix, before/after RMSD table, commit sha. If after genuine effort the
divergence resists (2-attempt budget INCLUDING this one counts as attempt 1), document precisely
and leave uncommitted.
</task>
<completeness_contract>
Done = first divergence named with evidence; fix committed w/ connected bit-identity + material
RMSD improvement on >=3 of 4 disconnected graphs; notes written; ogdf-src tree left git-clean;
scripts/ogdf_runner untouched.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r76/fmmm-disconnected only. Never modify scripts/ogdf_runner,
~/tools/ogdf-src (restore contract), eval code, or other engines.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices. Stop only for correctness-critical ambiguity.
</default_follow_through_policy>
