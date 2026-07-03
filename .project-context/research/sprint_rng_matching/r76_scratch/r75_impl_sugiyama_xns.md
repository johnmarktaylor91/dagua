<task>
Implement STAGE A of the graphviz-dot x-coordinate port for dagua's sugiyama graphviz-fidelity
variants: replace Brandes-Kopf x-assignment with graphviz's auxiliary-graph network-simplex
(LR-balance) for fidelity_mode=graphviz ONLY. APPROVED-WITH-CHANGES by the adversarial critique
(verdicts 14-16 in .project-context/research/sprint_rng_matching/r75_findings/
r75_ADVERSARIAL_VERDICTS.md -- read it, plus the two research reports
r75_sugiyama_codex.md (F1/F2 port spec + verification ladder) and r75_sugiyama_sonnet.md).

Repo/worktree: /home/jtaylor/.claude/worktrees/dagua-sugiyama-xns (branch r75/sugiyama-xns
@ 89ed3c3). Work ONLY here. Conventional commits (feat(sugiyama): ...). No push, no merge.

REFERENCE (version-pinned: run `git -C /home/jtaylor/projects/_references/graphviz show
7.0.5:<path>` -- the working tree is at a NEWER version, do NOT read it directly):
- 7.0.5:lib/dotgen/position.c -- dot_position flow (:120-135), create_aux_edges (:218-343),
  set_xcoords (:570-584).
- 7.0.5:lib/common/ns.c -- network simplex, balance=2 LR behavior.
- Aux-graph construction rules (from the F2 spec, verify each against 7.0.5 source):
  * same-rank neighbor constraint: aux edge tail=rank[i][j], head=rank[i][j+1],
    minlen=round(rw(left)+lw(right)+nodesep), weight=0.
  * per original/virtual edge e=(t,h): new slack node ne with two aux edges ne->t
    (minlen=max(port_dx,0)+1... verify exact 7.0.5 formula) and ne->h, weight = edge weight x
    omega table by endpoint types (C_EE=1, C_VS=2, C_SS=2, C_VV=4 -- verify at
    7.0.5:lib/dotgen/mincross.c virtual_weight).
  * STAGE A SCOPE: no clusters, no edge labels, no ports (port_dx=0), no flat-edge constraints.

DAGUA SIDE:
- Pipeline dispatch: dagua/layout/ops/pipelines/sugiyama.py:102-124 (graphviz fidelity currently
  falls through to _CoordinateAssignment = BK at dagua/layout/ops/sugiyama.py:1727-1738).
- Reuse the existing network-simplex backend used for ranking (dagua/layout/ops/dot_rank.py or
  equivalent -- find graphviz_rank_assignment's solver) -- it may need an LR-balance mode
  (balance=2 semantics: after optimality, move degree-balanced nodes to the median of allowed
  range... verify in 7.0.5:lib/common/ns.c).
- Add a new op (e.g. graphviz_xcoord_ns) invoked ONLY for fidelity_mode in {"graphviz"}; keep BK
  for igraph/default modes untouched. y-ranks unchanged.

VERIFICATION LADDER (from the research reports; run each via the benchmark path,
MPLCONFIGDIR=/tmp/mpl, cd worktree):
a. binary_tree: ranks/orders already match graphviz (proven in research). After your port, x
   coordinates must match graphviz's up to the known affine frame (translation/axis-sign/scale):
   compare against the saved reference positions in the MAIN repo
   /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_seeded_refs/positions/
   binary_tree__graphviz_dot__for__classic_sugiyama_graphviz_fidelity.pt (read-only) --
   compute the residual after optimal similarity alignment; target < 1e-6 relative.
   If it does not hit that, iterate: diff aux-graph edges/minlens/weights against a manual trace
   of 7.0.5 source on this 11-node tree until it does or you can name the exact remaining rule.
b. bipartite_4_3_4, org_chart_1_5_4_8, center_port_backedge_hub: stress gap vs reference must
   SHRINK materially vs the values in r75_targets_sugiyama.json (same alignment methodology).
c. REGRESSION GATE: run 5 seeds x {classic_sugiyama_default, classic_sugiyama_tight} (igraph
   family, must be BYTE-IDENTICAL to pre-change -- they must not route through your new op) and
   pytest tests/ -k sugiyama -x -q all green.
Write .project-context/research/sprint_rng_matching/r75_findings/r75_IMPL_sugiyama_xns_NOTES.md:
per ladder step, before/after numbers; residual rules not yet ported (labels/clusters/flat -> the
planned stages B-D); commits.
</task>
<completeness_contract>
Done = stage A lands gated to graphviz fidelity, ladder step (a) achieves frame-level x match on
binary_tree (or you name the precise unported rule blocking it with 7.0.5 line cites), (b) shows
material stress-gap shrink on >=2/3 graphs, (c) regression gate green, notes written. This is
stage A of a multi-stage port -- do not attempt clusters/labels/flat edges.
</completeness_contract>
<action_safety>
No pushes/merges. Commits on r75/sugiyama-xns only. Never modify igraph/default sugiyama paths,
eval metrics, or the main repo checkout.
</action_safety>
<default_follow_through_policy>
Most reasonable low-risk interpretation; note choices in NOTES. Stop only for correctness-critical
ambiguity.
</default_follow_through_policy>
